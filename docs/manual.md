# Glacier.Compute User Manual

## Contents

1. [Installation & Requirements](#1-installation--requirements)
2. [Shader Bytecode Assembler (`Glacier.Compute.Assembler`)](#2-shader-bytecode-assembler-glaciercomputeassembler)
3. [Shader Pipeline & Caching (`Glacier.Compute.Pipeline`)](#3-shader-pipeline--caching-glaciercomputepipeline)
4. [Multi-Tier Memory Sub-Allocators (`Glacier.Compute.Memory`)](#4-multi-tier-memory-sub-allocators-glaciercomputememory)
5. [Distributed Ring AllReduce (`Glacier.Compute.Distrib`)](#5-distributed-ring-allreduce-glaciercomputedistrib)
6. [Direct CUDA Driver Interop (`CudaDriver`)](#6-direct-cuda-driver-interop-cudadriver)
7. [Building & Testing](#7-building--testing)

---

## 1. Installation & Requirements

```bash
dotnet add package Glacier.Compute
```

- **Target framework:** `.NET 10` (`net10.0`) with `<IsAotCompatible>true</IsAotCompatible>`.
- **External dependencies:** Zero NuGet runtime dependencies. `CudaDriver` resolves `nvcuda.dll` (Windows) or `libcuda.so.1` (Linux) dynamically when present.

---

## 2. Shader Bytecode Assembler (`Glacier.Compute.Assembler`)

### IR Opcodes (`ComputeOpcode` & `Instruction`)

Each instruction is an immutable value struct:

```csharp
public readonly record struct Instruction(int Opcode, int OperandA, int OperandB, int Destination);
```

Instructions operate over 64 virtual 32-bit float registers (`f0..f63`) and 64 virtual 32-bit unsigned integer index registers (`r0..r63`):

| Category | `ComputeOpcode` Members |
| :--- | :--- |
| **Control & Indexing** | `Nop`, `LocalId`, `GlobalId`, `BarrierSync`, `Return` |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Fma`, `Neg`, `Abs`, `Min`, `Max` |
| **Transcendental** | `Sqrt`, `Rsqrt`, `Exp`, `Log`, `Tanh` |
| **Memory & Atomics** | `LoadGlobal`, `StoreGlobal`, `LoadShared`, `StoreShared`, `LoadConstant`, `AtomicAdd` |
| **Warp Collective** | `ShflSyncBfly` (`shfl.sync.bfly` / `OpGroupNonUniformShuffleXor`) |

### Emitting NVIDIA PTX 8.0 (`PtxEmitter` & `PtxKernelBuilder`)

Emit PTX 8.0 assembly directly from an `Instruction` span or compose custom PTX kernels fluent-style:

```csharp
using Glacier.Compute.Assembler.Ptx;

string ptx = PtxEmitter.Emit(instructions, kernelName: "vec_add", targetArch: "sm_89");

string customPtx = new PtxKernelBuilder("axpy_kernel")
    .SetTargetArch("sm_89")
    .AddParameter(".u64", "param_x")
    .AddParameter(".u64", "param_y")
    .AddParameter(".f32", "param_alpha")
    .SetRegisterCounts(b32: 16, b64: 8, f32: 16)
    .SetSharedMemoryBytes(4096)
    .Emit("mov.u32 %r0, %tid.x;")
    .EmitFma("%f2", "%f0", "%f1", "%f2")
    .EmitBarrier()
    .Build();
```

### Emitting & Validating SPIR-V 1.6 (`SpirvBinaryEmitter` & `SpirvValidator`)

```csharp
using Glacier.Compute;
using Glacier.Compute.Assembler.SpirV;

var emitter = new SpirvBinaryEmitter();
ReadOnlyMemory<byte> spirv = emitter.EmitModule(
    ShaderStage.Compute,
    instructions,
    localSizeX: 64,
    localSizeY: 1,
    localSizeZ: 1);

if (!SpirvValidator.Validate(spirv.Span, out string? error))
{
    throw new InvalidOperationException($"Invalid SPIR-V module: {error}");
}
```

- `EmitWords(stage, instructions, destinationWords, localSizeX: 64)` writes SPIR-V words directly into a caller-supplied `Span<uint>` with zero heap allocation for the output buffer.
- `SpirvBinaryEmitter` throws `NotSupportedException` if compute-only built-ins (`GlobalId`, `LocalId`, `LoadShared`, `StoreShared`, `BarrierSync`, `ShflSyncBfly`, `AtomicAdd`) are requested for `ShaderStage.Vertex` or `ShaderStage.Pixel`, or if an unrecognized opcode is supplied.

### Packaging Precompiled DXBC Containers (`DxbcContainerBuilder` & `DxbcContainerReader`)

Because retail Direct3D 12 drivers reject unsigned SM 6.0+ DXIL blobs unless compiled by `dxc` and signed by `dxil.dll`, direct IR-to-DXIL emission (`DxilChunkEmitter.CreateDxilChunk` / `ShaderBytecodeBuilder.BuildDxil`) throws `NotSupportedException`. Package precompiled LLVM 3.7 bitcode payloads produced by `dxc` instead:

```csharp
using Glacier.Compute;
using Glacier.Compute.Assembler.Dxbc;

ReadOnlyMemory<byte> dxbc = DxbcContainerBuilder.PackagePrecompiledShader(
    ShaderStage.Compute,
    precompiledBitcodeBytes,
    threadsX: 64,
    threadsY: 1,
    threadsZ: 1);

DxbcContainerReader parsed = DxbcContainerReader.Parse(dxbc.Span);
DxbcChunk? dxilChunk = parsed.FindChunk(DxbcChunk.FourCcDxil);
```

---

## 3. Shader Pipeline & Caching (`Glacier.Compute.Pipeline`)

`ShaderPipeline` compiles and caches `ComputeKernel` instances in a thread-safe bounded LRU cache (`DefaultMaxCachedKernels = 1024`):

```csharp
using Glacier.Compute;
using Glacier.Compute.Pipeline;

var pipeline = new ShaderPipeline(builder: null, maxCachedKernels: 512);

// includeDxil defaults to false (PTX 8.0 + SPIR-V 1.6)
ComputeKernel kernel = pipeline.Compile(
    "vector_add",
    ShaderStage.Compute,
    instructions,
    includePtx: true,
    includeSpirv: true,
    includeDxil: false);
```

Cache lookups key on `(ShaderStage, backendFlags, instructionSequence)` using alternate span lookup (`KernelLookupKey`), avoiding array allocations on cache hits.

---

## 4. Multi-Tier Memory Sub-Allocators (`Glacier.Compute.Memory`)

Every allocator supports two modes:
- **Standalone host mode:** Allocates unmanaged host memory arenas with no CPU-side page faulting on unused slabs.
- **External device/pinned arena mode:** Pass `(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)` so the allocator manages offsets solely in host metadata without dereferencing device pointers on the CPU.

### Unified Pool (`DeviceMemoryPool` / `IVramAllocator`)

```csharp
using Glacier.Compute;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Memory;

// External CUDA VRAM arena
if (CudaDriver.IsAvailable)
{
    CudaDriver.cuMemAlloc(out IntPtr vramPtr, 128UL * 1024 * 1024);
    using var gpuPool = new DeviceMemoryPool(vramPtr, 128UL * 1024 * 1024, p => CudaDriver.cuMemFree(p));

    DevicePointer small = gpuPool.Allocate(4096, MemoryCategory.SlabSmall);
    DevicePointer medium = gpuPool.Allocate(1024 * 1024, MemoryCategory.BuddyMedium);
    DevicePointer large = gpuPool.Allocate(16 * 1024 * 1024, MemoryCategory.LargeReservoir);

    using (ScratchWorkspace ws = gpuPool.RentScratchWorkspace(65536))
    {
        IntPtr tempDeviceAddr = ws.Pointer;
    }

    gpuPool.Free(large);
    gpuPool.Free(medium);
    gpuPool.Free(small);
}
```

When you call `pool.Allocate(ulong byteCount)` without a `MemoryCategory`, `DeviceMemoryPool` routes sizes automatically:
- `<= 64 KB` -> `SlabTier` (`SlabAllocator`)
- `<= 32 MB` -> `BuddyTier` (`BuddyAllocator`)
- `> 32 MB` -> `VirtualTier` (`VirtualMemoryReservoir`)

External arena construction requires at least `DeviceMemoryPool.MinimumExternalArenaBytes` (`1 MB`). Inspect counters via `SlabAllocationsCount`, `BuddyAllocationsCount`, and `VirtualAllocationsCount`.

### Individual Tier Allocators

| Allocator | Range / Behavior | Key Methods & Properties |
| :--- | :--- | :--- |
| **`SlabAllocator`** | 9 bins (`256 B` to `64 KB` in `SlabAllocator.BinSizes`) | `Allocate(bytes)`, `Free(ptr)`, `Contains(addr)`, `TotalCommittedBytes`, `ActiveAllocatedBytes` |
| **`BuddyAllocator`** | `128 KB` (`MinBlockSize`) to `32 MB` (`MaxBlockSize`, `9` orders) | `Allocate(bytes)`, `Free(ptr)`, `Contains(addr)`, `GetOrderForSize`, `GetBlockSize`, `AllocatedBytes` |
| **`VirtualMemoryReservoir`** | Zero-RSS virtual reserve (`VirtualAlloc`/`mmap`) | `Allocate(bytes)`, `Free(ptr)`, `Contains(addr)`, `TotalReservedBytes`, `TotalCommittedBytes` |
| **`RingScratchWorkspace`** | Lock-free circular scratch bump (`64 B` aligned) | `Allocate`, `TryAllocate` (with optional `TimeSpan timeout`), `Free`, `Contains`, `CurrentOffset` |

---

## 5. Distributed Ring AllReduce (`Glacier.Compute.Distrib`)

`DistributedContext` runs two-phase ring AllReduce (`ReductionOp`: `Sum`, `Average`, `Min`, `Max`) using `SimdAccumulator`:

```csharp
using Glacier.Compute;
using Glacier.Compute.Distrib;

// In-memory 4-rank ring
IDistributedContext[] cluster = DistributedAllReduce.CreateLocalCluster(worldSize: 4);
try
{
    var buffers = new Memory<float>[4];
    for (int r = 0; r < 4; r++)
    {
        float[] data = new float[65536];
        Array.Fill(data, r + 1.0f);
        buffers[r] = data;
    }

    await DistributedAllReduce.ExecuteClusterAsync(cluster, buffers, ReductionOp.Sum);
}
finally
{
    foreach (var ctx in cluster) ctx.Dispose();
}
```

For multi-process or multi-host TCP clusters, wrap connected `Socket` pairs with `DistributedAllReduce.CreateSocketContext(rank, worldSize, successorSocket, predecessorSocket)` or construct `RingSocketTransport` directly (enforcing `DefaultMaxFrameBytes = 64 MB` per message frame).

---

## 6. Direct CUDA Driver Interop (`CudaDriver`)

`Glacier.Compute.Assembler.Ptx.CudaDriver` exposes source-generated Native AOT wrappers around `Glacier.Compute.Drivers.CuDriver`:

- Check availability via `CudaDriver.IsAvailable` before calling `cuInit`, `cuDeviceGet`, `cuCtxCreate`, `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuCtxSynchronize`, or `cuCtxDestroy`.

---

## 7. Building & Testing

```bash
dotnet build Glacier.Compute.slnx -c Release
dotnet test Glacier.Compute.slnx -c Release --no-build
```

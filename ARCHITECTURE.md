# Glacier.Compute Architecture (`v1.1.0`)

## 1. Design Principles

1. **Zero Hidden Native Dependencies**: `Glacier.Compute` ships as a pure C# `.NET 10` assembly (`<IsAotCompatible>true</IsAotCompatible>`, `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`) with zero NuGet runtime dependencies.
2. **Honest Hardware Boundaries**:
   - Backends that can be validly emitted in pure C# (`PTX 8.0` assembly text and `SPIR-V 1.6` binary words) are generated directly in memory and validated structurally.
   - Backends requiring proprietary driver-enforced cryptographic signing (`Direct3D 12` SM 6.0+ `DXIL` via `dxil.dll`) explicitly throw `NotSupportedException` on raw IR synthesis while supporting DXBC container packaging (`DxbcContainerBuilder.PackagePrecompiledShader`) for precompiled DXC bitcode payloads.
3. **Decoupled Arena Sub-Allocation**: Memory sub-allocators (`SlabAllocator`, `BuddyAllocator`, `VirtualMemoryReservoir`, `RingScratchWorkspace`, and `DeviceMemoryPool`) maintain all bookkeeping in host metadata structures and **never dereference arena addresses on the CPU**, allowing identical ~15–70 ns sub-allocation over GPU VRAM pointers (`CUdeviceptr`, `D3D12_GPU_VIRTUAL_ADDRESS`), pinned DMA host buffers, or standalone unmanaged host arenas.

---

## 2. Shader Bytecode Assembler (`Glacier.Compute.Assembler`)

### 2.1 Intermediate Representation (`Instruction` & `ComputeOpcode`)
Kernels are expressed as flat `ReadOnlySpan<Instruction>` streams over 64 virtual 32-bit float registers (`f0..f63`) and 64 virtual 32-bit unsigned integer index registers (`r0..r63`), covering 26 opcodes (`Nop`, `Add`, `Sub`, `Mul`, `Div`, `Fma`, `Neg`, `Abs`, `Min`, `Max`, `Sqrt`, `Rsqrt`, `Exp`, `Log`, `Tanh`, `LoadGlobal`, `StoreGlobal`, `LoadShared`, `StoreShared`, `LoadConstant`, `ShflSyncBfly`, `BarrierSync`, `LocalId`, `GlobalId`, `AtomicAdd`, `Return`).

### 2.2 NVIDIA PTX 8.0 Emitter (`PtxEmitter` & `CudaDriver`)
- Emits `.version 8.0`, `.target sm_90` (configurable), `.address_size 64` compute kernels with 4 standard ABI parameters: `param_input_a` (`.u64`), `param_input_b` (`.u64`), `param_output` (`.u64`), and `param_n` (`.u32`), plus 4 KB of `.shared` workspace (`smem_workspace[4096]`).
- `CudaDriver` resolves `nvcuda.dll` / `libcuda.so` exports lazily via `NativeLibrary.TryLoad` into unmanaged function pointers (`delegate* unmanaged<...>`) using stack-allocated UTF-8 marshaling for full Native AOT compatibility.

### 2.3 SPIR-V 1.6 Binary Emitter & Validator (`SpirvBinaryEmitter` & `SpirvValidator`)
`SpirvBinaryEmitter` emits binary modules strictly adhering to Section 2.4 of the Khronos SPIR-V 1.6 specification:
1. **Header**: Magic `0x07230203`, Version `0x00010600` (1.6), Generator `0x00130000`, `Bound`, Schema `0`.
2. **Capabilities & Extensions**: Always declares `CapabilityShader` (`1`). Conditionally declares `CapabilityGroupNonUniformShuffle` (`65`) when `ShflSyncBfly` is used, and `CapabilityAtomicFloat32AddEXT` (`6033`) + `OpExtension "SPV_EXT_shader_atomic_float_add"` when `AtomicAdd` is used.
3. **Extended Instructions**: Imports `GLSL.std.450` (`OpExtInstImport`) to lower `Fma` (`50`), `FAbs` (`4`), `FMin` (`37`), `FMax` (`40`), `Sqrt` (`31`), `InverseSqrt` (`32`), `Exp` (`27`), `Log` (`28`), and `Tanh` (`21`).
4. **Descriptor & Interface Layout (`ShaderStage.Compute`)**:
   - `set = 0, binding = 0`: `StorageBuffer` (`input_a`, `float[]`, `ArrayStride = 4`)
   - `set = 0, binding = 1`: `StorageBuffer` (`input_b`, `float[]`, `ArrayStride = 4`)
   - `set = 0, binding = 2`: `StorageBuffer` (`output`, `float[]`, `ArrayStride = 4`)
   - `PushConstant`: `{ uint param_n; }` (`Block`, member `Offset = 0`)
   - `Input` built-ins: `BuiltInGlobalInvocationId` and `BuiltInLocalInvocationId` (strictly omitted for `Vertex` and `Pixel` stages)
   - `Workgroup` shared array: `float[1024]` (4096 bytes), emitted when `LoadShared` or `StoreShared` is present
   - All module-scope `OpVariable` IDs are listed on `OpEntryPoint` per SPIR-V 1.4+ / 1.6 rules.
5. **Control Flow & Validation**:
   - Pre-scans all instructions so `OpType*`, `OpConstant`, and global `OpVariable` declarations precede `OpFunction`, and allocates local `StorageClassFunction` variables solely for live registers at the start of the entry block.
   - Tracks basic-block termination (`terminated`) so `OpReturn` is emitted strictly once.
   - Throws `NotSupportedException` for unknown opcodes or compute-only opcodes used in `Vertex`/`Pixel` stages.
   - `SpirvValidator.Validate` verifies word framing, `0 < id < bound` across instruction ID operands, and single basic-block termination before `OpFunctionEnd`.

### 2.4 DXBC Container Packager (`DxbcContainerBuilder` & `DxbcContainerReader`)
- Packages precompiled DXC LLVM 3.7 bitcode (`0xDEC04342` / `0x0B17C0DE`) into a 16-byte `DXIL` program header chunk alongside `PSV0` metadata chunks inside a `DXBC` container.
- Note on D3D12 signing: Retail Windows D3D12 drivers require the proprietary `dxil.dll` validator hash unless Windows Developer Mode (`D3D12EnableExperimentalFeatures`) is enabled. `DxbcContainerBuilder` writes an unsigned MD5 digest over container bytes `[20..end]` for structural verification; direct IR-to-DXIL synthesis is gated with `NotSupportedException`.

---

## 3. Multi-Tier Arena Sub-Allocator (`Glacier.Compute.Memory`)

### 3.1 Backing Modes
Every memory allocator provides two construction paths:
1. **External Device / Pinned Arena (`(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction)`)**:
   - Sub-allocates caller-managed GPU VRAM (`cuMemAlloc`, `ID3D12Heap`, `VkDeviceMemory`) or pinned host memory (`cuMemHostAlloc`).
   - `DeviceMemoryPool(baseAddress, byteLength, releaseAction)` partitions the external range into 64 KB-aligned non-overlapping sub-arenas: ~12.5% `SlabTier`, ~12.5% `ScratchTier`, ~50% `BuddyTier`, and ~25% `VirtualTier`, invoking `releaseAction(baseAddress)` once upon `Dispose()`.
2. **Standalone Unmanaged Host Arena (default constructors)**:
   - `SlabAllocator`: Lazily allocates 64-byte-aligned unmanaged bins on first allocation per bin size (zero upfront RSS for unused bins).
   - `BuddyAllocator`: Allocates a 4 KB-aligned unmanaged host arena without upfront full-arena zero-faulting; decomposes capacity into power-of-two free lists (128 KB – 32 MB) and recursively coalesces buddies within arena bounds.
   - `VirtualMemoryReservoir`: Reserves address space with **zero physical RSS** via `VirtualAlloc(..., MEM_RESERVE, PAGE_NOACCESS)` on Windows and `mmap(..., PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)` on Linux (`0x20`) and macOS/BSD (`0x1000`). Pages are committed on `Allocate` (`MEM_COMMIT` / `mprotect(PROT_READ | PROT_WRITE)`) and decommitted on `Free` (`MEM_DECOMMIT` / `madvise(MADV_DONTNEED)` + `mprotect(PROT_NONE)`).
   - `RingScratchWorkspace`: Uses a single lock-free `Interlocked.CompareExchange` loop computing `pos = current % capacity`, wrap padding `pad`, and `next = current + pad + alignedSize` atomically so concurrent threads never receive overlapping slices or desynchronize ring offsets.

---

## 4. Distributed Ring AllReduce (`Glacier.Compute.Distrib`)

- **Scatter-Reduce & AllGather (`DistributedContext`)**: Splits a float buffer of length $M$ into $N$ balanced chunks (`baseChunk = M / N`, first `M % N` chunks receive `+1` element). Runs $N-1$ concurrent send/receive steps accumulating via `SimdAccumulator`, followed by $N-1$ ring broadcast steps.
- **SIMD Vectorization (`SimdAccumulator`)**: Dispatches at JIT/AOT compile time to `Vector512<float>`, `Vector256<float>`, or `Vector128<float>` loops with scalar tail cleanup for `Sum`, `Average`, `Min`, and `Max`.

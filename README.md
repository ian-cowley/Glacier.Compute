![Glacier.Compute Banner](assets/banner.jpg)

# Glacier.Compute

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)

`Glacier.Compute` is a pure C# shader bytecode assembler, multi-tier memory sub-allocator, and ring AllReduce library for .NET 10.

Use it when your .NET application needs to emit GPU compute shaders in memory without launching `nvcc` or `glslc`, sub-allocate pre-allocated VRAM or host arenas in tens of nanoseconds instead of calling `cuMemAlloc` per tensor, or synchronize float buffers across CPU or socket ranks with SIMD vector reductions.

## Install

```bash
dotnet add package Glacier.Compute
```

## Quick Start

```csharp
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Memory;
using Glacier.Compute.Pipeline;

// 1. Assemble PTX 8.0 and SPIR-V 1.6 bytecode in memory
var pipeline = new ShaderPipeline();
var instructions = new Instruction[]
{
    new((int)ComputeOpcode.GlobalId, 0, 0, 0),
    new((int)ComputeOpcode.LoadGlobal, 0, 0, 1),
    new((int)ComputeOpcode.LoadGlobal, 1, 0, 2),
    new((int)ComputeOpcode.Add, 1, 2, 3),
    new((int)ComputeOpcode.StoreGlobal, 3, 0, 0),
    new((int)ComputeOpcode.Return, 0, 0, 0)
};

ComputeKernel kernel = pipeline.Compile("vector_add", ShaderStage.Compute, instructions);
string ptx = kernel.PtxAssembly!;
ReadOnlyMemory<byte> spirv = kernel.SpirvBytecode!.Value;

// 2. Sub-allocate an unmanaged host pool or an external GPU VRAM arena
using var pool = new DeviceMemoryPool(
    buddyArenaBytes: 16 * 1024 * 1024,
    virtualReserveBytes: 256 * 1024 * 1024,
    scratchBytes: 4 * 1024 * 1024);

DevicePointer block = pool.Allocate(4096, MemoryCategory.SlabSmall);
pool.Free(block);

using ScratchWorkspace scratch = pool.RentScratchWorkspace(65536);
IntPtr scratchPtr = scratch.Pointer;
```

## What You Can Do

| Subsystem | Capabilities | Key Types |
| :--- | :--- | :--- |
| **In-Memory Shader Assembler** | Emits PTX 8.0, validated SPIR-V 1.6 compute modules, DXBC packaging for precompiled DXIL, and AOT CUDA driver calls | `ShaderPipeline`, `PtxEmitter`, `PtxKernelBuilder`, `SpirvBinaryEmitter`, `SpirvValidator`, `DxbcContainerBuilder`, `CudaDriver` |
| **Multi-Tier Arena Allocator** | Sub-allocates caller-owned VRAM or unmanaged host memory across slabs, buddy blocks, virtual pages, and CAS scratch rings | `DeviceMemoryPool`, `SlabAllocator`, `BuddyAllocator`, `VirtualMemoryReservoir`, `RingScratchWorkspace`, `ScratchWorkspace` |
| **Distributed Ring AllReduce** | Two-phase scatter-reduce and all-gather over in-memory channels or TCP sockets with AVX-512/AVX2/NEON accumulation | `DistributedAllReduce`, `DistributedContext`, `SimdAccumulator`, `InMemoryRingTransport`, `RingSocketTransport` |

## Platform & Compatibility

- Targets `.NET 10` (`<IsAotCompatible>true</IsAotCompatible>`, `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`) with zero third-party package dependencies.
- `PtxEmitter`, `SpirvBinaryEmitter`, `DeviceMemoryPool`, and `DistributedAllReduce` run on Windows and Linux without an installed GPU SDK.
- `CudaDriver` dynamically resolves `nvcuda.dll` or `libcuda.so.1` via native function pointers (`delegate* unmanaged`).
- Retail Direct3D 12 drivers require external DXC compilation and `dxil.dll` validator signing, so direct IR-to-DXIL synthesis (`DxilChunkEmitter.CreateDxilChunk` / `ShaderBytecodeBuilder.BuildDxil`) throws `NotSupportedException` and `ShaderPipeline.Compile` defaults `includeDxil: false`.

## Benchmarks

Measured with `dotnet run --project benchmarks/Glacier.Compute.Benchmarks/Glacier.Compute.Benchmarks.csproj -c Release -- --quick` on an AMD Ryzen AI 9 HX 370 (`.NET 10.0.15`, Windows 11):

| Operation | Measured Result | Notes |
| :--- | ---: | :--- |
| `PtxEmitter.Emit` (32 instructions) | **3.85 μs** (`259.6K ops/s`) | Pure C# PTX 8.0 text generation |
| `SpirvBinaryEmitter.EmitModule` (32 instructions) | **2.91 μs** (`343.9K ops/s`) | Spec-compliant SPIR-V 1.6 words |
| `DxbcContainerBuilder.PackagePrecompiledShader` | **854.6 ns** (`1.17M ops/s`) | Precompiled DXIL + `PSV0` container |
| `SlabAllocator` (1 KB alloc + free) | **68.85 ns** | Lock-free 64-bit bitmask CAS |
| `BuddyAllocator` (256 KB alloc + free) | **521.48 ns** | Recursive companion coalescing |
| `VirtualMemoryReservoir` (1 MB commit + free) | **5.07 μs** | Zero-RSS reserve, on-demand commit |
| `SimdAccumulator.Accumulate` (64K floats / 256 KB) | **3.24 μs** (`161.68 GB/s`) | Vectorized `Sum` reduction |
| `DistributedAllReduce` (4-node ring, 256 KB) | **0.243 ms** / step | `InMemoryRingTransport` scatter + gather |

See [docs/benchmarks.md](docs/benchmarks.md) for the full run details and BenchmarkDotNet suites.

## Documentation

- [User Manual](docs/manual.md) — shader IR opcodes, external VRAM sub-allocation, all four memory tiers, and TCP/in-memory ring AllReduce
- [Architecture](docs/architecture.md) — SPIR-V 1.6 binary framing, DXBC constraints, external arena partitioning, and transport design
- [Benchmarks](docs/benchmarks.md) — full hardware/runtime spec and reproducible microbenchmark runs
- [Changelog](CHANGELOG.md) — release notes

## License & Credits

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Ian Cowley.

Developed by Ian Cowley, Antigravity (Google DeepMind) and Claude.

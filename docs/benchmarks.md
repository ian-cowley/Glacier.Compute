# Glacier.Compute Benchmarks

All measurements below were collected on physical hardware with zero simulated steps.

## Test Environment

- **CPU:** AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, Zen 5, AVX-512)
- **RAM:** 31 GB
- **OS:** Windows 11 (`10.0.26300.0`)
- **Runtime:** `.NET 10.0.15` (`.NET SDK 11.0.100`, Release x64)

## Reproduce Locally

```bash
# Quick diagnostic suite
dotnet run --project benchmarks/Glacier.Compute.Benchmarks/Glacier.Compute.Benchmarks.csproj -c Release -- --quick

# Full BenchmarkDotNet suite (AllocatorBenchmarks, AllReduceBenchmarks, AssemblerBenchmarks)
dotnet run --project benchmarks/Glacier.Compute.Benchmarks/Glacier.Compute.Benchmarks.csproj -c Release -- --full
```

---

## Measured Results (`--quick`, Release)

### 1. In-Memory Shader Bytecode Assembler (`10,000` iterations, 32-instruction compute kernel)

| Benchmark | Measured Latency | Throughput |
| :--- | ---: | ---: |
| `PtxEmitter.Emit` (PTX 8.0 JIT text) | **3,852.53 ns / kernel** | `259.57K ops/sec` |
| `SpirvBinaryEmitter.EmitModule` (SPIR-V 1.6 words) | **2,907.62 ns / module** | `343.92K ops/sec` |
| `DxbcContainerBuilder.PackagePrecompiledShader` | **854.63 ns / container** | `1,170.10K ops/sec` |
| `ShaderPipeline.Compile` (cached lookup) | **1,834.32 ns / lookup** | `545.16K ops/sec` |

### 2. Multi-Tier Arena Sub-Allocators

| Tier / Operation | Iterations | Measured Latency | Notes |
| :--- | :---: | ---: | :--- |
| `SlabAllocator` (1 KB alloc + free) | 50,000 | **68.85 ns / cycle** | 0 driver calls; lock-free bitmask CAS |
| `BuddyAllocator` (256 KB alloc + free) | 50,000 | **521.48 ns / cycle** | Recursive companion coalescing |
| `VirtualMemoryReservoir` (1 MB commit + free) | 2,000 | **5.07 μs / cycle** | Zero-RSS reserve, OS page commit/decommit |
| `RingScratchWorkspace` (4 KB pointer bump) | 50,000 | **139.58 ns / bump** | Lock-free CAS wrap-safe ring |

### 3. SIMD Vector Accumulation & Ring AllReduce

| Benchmark | Scope | Iterations | Measured Result |
| :--- | :--- | :---: | :--- |
| `SimdAccumulator.Accumulate` (`Sum`) | `65,536` floats (`256 KB`) | 10,000 | **3.24 μs** (**161.68 GB/s**) |
| `DistributedAllReduce.ExecuteClusterAsync` | 4-node in-memory ring, `256 KB` | 200 | **0.243 ms / step** |

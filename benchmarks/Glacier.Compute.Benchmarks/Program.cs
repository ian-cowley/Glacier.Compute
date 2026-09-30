// <copyright file="Program.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Benchmarks;

using System;
using System.Diagnostics;
using System.Threading.Tasks;
using BenchmarkDotNet.Running;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Assembler.SpirV;
using Glacier.Compute.Distrib;
using Glacier.Compute.Memory;
using Glacier.Compute.Pipeline;

public static class Program
{
    public static async Task Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--quick")
        {
            await RunQuickDiagnosticsAsync();
            return;
        }

        Console.WriteLine("===============================================================================");
        Console.WriteLine("Glacier.Compute High-Performance Microbenchmarks (.NET 10)");
        Console.WriteLine("===============================================================================");
        Console.WriteLine("Running quick diagnostic benchmark suite...");
        await RunQuickDiagnosticsAsync();

        if (args.Length > 0 && args[0] == "--full")
        {
            Console.WriteLine("Running full BenchmarkDotNet suite...");
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
        }
    }

    private static async Task RunQuickDiagnosticsAsync()
    {
        Console.WriteLine();
        Console.WriteLine("-------------------------------------------------------------------------------");
        Console.WriteLine("1. In-Memory Shader Bytecode Assembler Latency");
        Console.WriteLine("-------------------------------------------------------------------------------");

        var instructions = new Instruction[32];
        for (int i = 0; i < 32; i++)
        {
            instructions[i] = new Instruction((int)ComputeOpcode.Add, i, (i + 1) % 32, (i + 2) % 32);
        }

        var spirvEmitter = new SpirvBinaryEmitter();
        var pipeline = new ShaderPipeline();

        // Warmup
        for (int i = 0; i < 1000; i++)
        {
            _ = PtxEmitter.Emit(instructions);
            _ = spirvEmitter.EmitModule(ShaderStage.Compute, instructions);
            _ = DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions);
            _ = pipeline.Compile("w", ShaderStage.Compute, instructions);
        }

        const int iterations = 10_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = PtxEmitter.Emit(instructions);
        }
        sw.Stop();
        double ptxNs = (sw.Elapsed.TotalNanoseconds) / iterations;
        Console.WriteLine($"  [PTX 8.0 JIT Generator]       : {ptxNs:F2} ns / kernel ({1_000_000.0 / ptxNs:F2}K ops/sec)");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = spirvEmitter.EmitModule(ShaderStage.Compute, instructions);
        }
        sw.Stop();
        double spirvNs = (sw.Elapsed.TotalNanoseconds) / iterations;
        Console.WriteLine($"  [SPIR-V 1.6 Binary Emitter]   : {spirvNs:F2} ns / module ({1_000_000.0 / spirvNs:F2}K ops/sec)");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions);
        }
        sw.Stop();
        double dxbcNs = (sw.Elapsed.TotalNanoseconds) / iterations;
        Console.WriteLine($"  [DXBC/DXIL Container Builder] : {dxbcNs:F2} ns / container ({1_000_000.0 / dxbcNs:F2}K ops/sec)");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = pipeline.Compile("cached", ShaderStage.Compute, instructions);
        }
        sw.Stop();
        double pipelineNs = (sw.Elapsed.TotalNanoseconds) / iterations;
        Console.WriteLine($"  [Pipeline Cached JIT Lookup]  : {pipelineNs:F2} ns / lookup ({1_000_000.0 / pipelineNs:F2}K ops/sec)");

        Console.WriteLine();
        Console.WriteLine("-------------------------------------------------------------------------------");
        Console.WriteLine("2. Unified Multi-Tier VRAM Allocator Latency");
        Console.WriteLine("-------------------------------------------------------------------------------");

        using var slab = new SlabAllocator(slotsPerBin: 1024);
        using var buddy = new BuddyAllocator(64 * 1024 * 1024);
        using var ring = new RingScratchWorkspace(64 * 1024 * 1024);

        // Warmup
        for (int i = 0; i < 5000; i++)
        {
            var p1 = slab.Allocate(1024);
            slab.Free(p1);
            var p2 = buddy.Allocate(256 * 1024);
            buddy.Free(p2);
            _ = ring.Allocate(4096);
        }

        const int allocIters = 50_000;
        sw.Restart();
        for (int i = 0; i < allocIters; i++)
        {
            var p = slab.Allocate(1024);
            slab.Free(p);
        }
        sw.Stop();
        double slabNs = (sw.Elapsed.TotalNanoseconds) / allocIters;
        Console.WriteLine($"  [Tier 1: Slab Alloc + Free]   : {slabNs:F2} ns / cycle (0 driver calls)");

        sw.Restart();
        for (int i = 0; i < allocIters; i++)
        {
            var p = buddy.Allocate(256 * 1024);
            buddy.Free(p);
        }
        sw.Stop();
        double buddyNs = (sw.Elapsed.TotalNanoseconds) / allocIters;
        Console.WriteLine($"  [Tier 2: Buddy Alloc + Merge] : {buddyNs:F2} ns / cycle (zero fragmentation)");

        sw.Restart();
        for (int i = 0; i < allocIters; i++)
        {
            _ = ring.Allocate(4096);
        }
        sw.Stop();
        double ringNs = (sw.Elapsed.TotalNanoseconds) / allocIters;
        Console.WriteLine($"  [Stream Scratch Pointer Bump] : {ringNs:F2} ns / bump (lock-free atomic)");

        Console.WriteLine();
        Console.WriteLine("-------------------------------------------------------------------------------");
        Console.WriteLine("3. SIMD Accumulation & Distributed Ring AllReduce");
        Console.WriteLine("-------------------------------------------------------------------------------");

        const int tensorSize = 65536; // 64K floats = 256 KB
        float[] dst = new float[tensorSize];
        float[] src = new float[tensorSize];
        Array.Fill(dst, 1.0f);
        Array.Fill(src, 2.0f);

        // Warmup
        for (int i = 0; i < 1000; i++)
        {
            SimdAccumulator.Accumulate(dst, src, ReductionOp.Sum);
        }

        const int simdIters = 10_000;
        sw.Restart();
        for (int i = 0; i < simdIters; i++)
        {
            SimdAccumulator.Accumulate(dst, src, ReductionOp.Sum);
        }
        sw.Stop();
        double simdUs = (sw.Elapsed.TotalMicroseconds) / simdIters;
        double throughputGbps = ((double)tensorSize * sizeof(float) * 2 / 1e9) / (simdUs / 1e6);
        Console.WriteLine($"  [SIMD Vector Accumulate 64K]  : {simdUs:F2} μs ({throughputGbps:F2} GB/s bandwidth)");

        var cluster4 = DistributedAllReduce.CreateLocalCluster(4);
        var buffers4 = new Memory<float>[4];
        for (int i = 0; i < 4; i++)
        {
            buffers4[i] = new float[tensorSize];
        }

        // Warmup
        for (int i = 0; i < 50; i++)
        {
            await DistributedAllReduce.ExecuteClusterAsync(cluster4, buffers4, ReductionOp.Sum);
        }

        const int ringIters = 200;
        sw.Restart();
        for (int i = 0; i < ringIters; i++)
        {
            await DistributedAllReduce.ExecuteClusterAsync(cluster4, buffers4, ReductionOp.Sum);
        }
        sw.Stop();
        double ringMs = (sw.Elapsed.TotalMilliseconds) / ringIters;
        Console.WriteLine($"  [Ring AllReduce 4-Node 256KB] : {ringMs:F3} ms / step");
        Console.WriteLine("===============================================================================");
    }
}

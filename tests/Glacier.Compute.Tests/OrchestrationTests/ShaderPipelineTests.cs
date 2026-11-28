// <copyright file="ShaderPipelineTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.OrchestrationTests;

using System;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.SpirV;
using Glacier.Compute.Pipeline;
using Xunit;

public class ShaderPipelineTests
{
    [Fact]
    public void ShaderPipeline_CompilesPtxAndSpirvByDefault_AndCachesKernel()
    {
        var pipeline = new ShaderPipeline();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        // First compile - builds PTX + SPIR-V (includeDxil defaults to false) and caches
        var kernel1 = pipeline.Compile("vector_add", ShaderStage.Compute, instructions);
        Assert.NotNull(kernel1);
        Assert.NotNull(kernel1.PtxAssembly);
        Assert.NotNull(kernel1.SpirvBytecode);
        Assert.Null(kernel1.DxilContainer);
        Assert.True(SpirvValidator.Validate(kernel1.SpirvBytecode.Value.Span, out string? err), err);
        Assert.Equal(1, pipeline.CachedKernelCount);

        // Second compile with identical instructions - returns cached instance
        var kernel2 = pipeline.Compile("vector_add", ShaderStage.Compute, instructions);
        Assert.Same(kernel1, kernel2);
        Assert.Equal(1, pipeline.CachedKernelCount);

        // Requesting direct DXIL synthesis throws honest NotSupportedException
        Assert.Throws<NotSupportedException>(() =>
            pipeline.Compile("vector_add_dxil", ShaderStage.Compute, instructions, includeDxil: true));
    }

    [Fact]
    public void ShaderPipeline_ValidatesCapacity_AndEvictsLeastRecentlyUsed()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ShaderPipeline(null, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ShaderPipeline(null, -4));

        var pipeline = new ShaderPipeline(builder: null, maxCachedKernels: 2);
        Assert.Equal(2, pipeline.MaxCachedKernels);

        var k1 = new Instruction[] { new((int)ComputeOpcode.Add, 0, 1, 2), new((int)ComputeOpcode.Return, 0, 0, 0) };
        var k2 = new Instruction[] { new((int)ComputeOpcode.Sub, 0, 1, 2), new((int)ComputeOpcode.Return, 0, 0, 0) };
        var k3 = new Instruction[] { new((int)ComputeOpcode.Mul, 0, 1, 2), new((int)ComputeOpcode.Return, 0, 0, 0) };

        var c1 = pipeline.Compile("k1", ShaderStage.Compute, k1);
        var c2 = pipeline.Compile("k2", ShaderStage.Compute, k2);
        Assert.Equal(2, pipeline.CachedKernelCount);

        // Touch k1 (different kernel name still hits same instruction cache key) so k2 becomes LRU
        var c1Hit = pipeline.Compile("k1_alias", ShaderStage.Compute, k1);
        Assert.Same(c1, c1Hit);

        // Insert k3 -> evicts k2, preserves k1
        var c3 = pipeline.Compile("k3", ShaderStage.Compute, k3);
        Assert.Equal(2, pipeline.CachedKernelCount);
        Assert.Same(c1, pipeline.Compile("k1_again", ShaderStage.Compute, k1));
        Assert.Same(c3, pipeline.Compile("k3_again", ShaderStage.Compute, k3));

        // Recompiling k2 must produce a fresh ComputeKernel instance since k2 was evicted
        var c2Recompiled = pipeline.Compile("k2", ShaderStage.Compute, k2);
        Assert.NotSame(c2, c2Recompiled);
    }

    [Fact]
    public async Task ShaderPipeline_ConcurrentCompilation_RespectsBoundedLruCapacity()
    {
        var pipeline = new ShaderPipeline(builder: null, maxCachedKernels: 8);

        var tasks = Enumerable.Range(0, 8).Select(tid => Task.Run(() =>
        {
            for (int i = 0; i < 40; i++)
            {
                int reg = (tid * 7 + i) % 32;
                var insts = new Instruction[]
                {
                    new((int)ComputeOpcode.Add, reg, (reg + 1) % 32, (reg + 2) % 32),
                    new((int)ComputeOpcode.Return, 0, 0, 0)
                };
                var kernel = pipeline.Compile($"k_{tid}_{i}", ShaderStage.Compute, insts);
                Assert.NotNull(kernel.PtxAssembly);
                Assert.NotNull(kernel.SpirvBytecode);
            }
        })).ToArray();

        await Task.WhenAll(tasks);
        Assert.InRange(pipeline.CachedKernelCount, 1, 8);
    }
}

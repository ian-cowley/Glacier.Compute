// <copyright file="ShaderPipelineTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.OrchestrationTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Pipeline;
using Xunit;

public class ShaderPipelineTests
{
    [Fact]
    public void ShaderPipeline_CompilesAndCachesKernel()
    {
        var pipeline = new ShaderPipeline();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        // First compile - builds and caches
        var kernel1 = pipeline.Compile("vector_add", ShaderStage.Compute, instructions);
        Assert.NotNull(kernel1);
        Assert.NotNull(kernel1.PtxAssembly);
        Assert.NotNull(kernel1.SpirvBytecode);
        Assert.NotNull(kernel1.DxilContainer);
        Assert.Equal(1, pipeline.CachedKernelCount);

        // Second compile with identical instructions - returns cached instance
        var kernel2 = pipeline.Compile("vector_add", ShaderStage.Compute, instructions);
        Assert.Same(kernel1, kernel2);
        Assert.Equal(1, pipeline.CachedKernelCount);
    }
}

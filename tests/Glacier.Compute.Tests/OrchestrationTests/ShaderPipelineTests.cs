// <copyright file="ShaderPipelineTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.OrchestrationTests;

using System;
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
}

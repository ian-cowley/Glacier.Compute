// <copyright file="ExtendedAssemblerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Assembler.SpirV;
using Xunit;

public class ExtendedAssemblerTests
{
    [Fact]
    public void ShaderBytecodeBuilder_ImplementsIShaderBytecodeBuilder_AllBackends()
    {
        IShaderBytecodeBuilder builder = new ShaderBytecodeBuilder();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        // PTX
        string ptx = builder.BuildPtx(instructions);
        Assert.Contains(".version 8.0", ptx);
        Assert.Contains("add.f32", ptx);

        // SPIR-V
        var spirv = builder.BuildSpirV(ShaderStage.Compute, instructions);
        Assert.True(SpirvValidator.Validate(spirv.Span, out _));

        // DXIL / DXBC
        var dxbc = builder.BuildDxil(ShaderStage.Compute, instructions);
        var reader = DxbcContainerReader.Parse(dxbc.Span);
        Assert.NotNull(reader.FindChunk(DxbcChunk.FourCcDxil));
    }

    [Fact]
    public void SpirvBinaryEmitter_VertexAndPixelStages()
    {
        var emitter = new SpirvBinaryEmitter();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var vertBytes = emitter.EmitModule(ShaderStage.Vertex, instructions);
        Assert.True(SpirvValidator.Validate(vertBytes.Span, out _));

        var pixelBytes = emitter.EmitModule(ShaderStage.Pixel, instructions);
        Assert.True(SpirvValidator.Validate(pixelBytes.Span, out _));
    }

    [Fact]
    public void DxbcContainerBuilder_CustomChunks()
    {
        var builder = new DxbcContainerBuilder();
        builder.AddChunk(new DxbcChunk("TEST", new byte[] { 1, 2, 3, 4 }));
        builder.AddChunk(new DxbcChunk("DATA", new byte[] { 5, 6, 7, 8, 9, 10 }));

        byte[] raw = builder.Build();
        var reader = DxbcContainerReader.Parse(raw);

        Assert.Equal(2, reader.Chunks.Count);
        var chunk1 = reader.FindChunk(BitConverter.ToUInt32(System.Text.Encoding.ASCII.GetBytes("TEST"), 0));
        Assert.NotNull(chunk1);
        Assert.Equal(4, chunk1.Data.Length);
        Assert.Equal(1, chunk1.Data[0]);

        var chunk2 = reader.FindChunk(BitConverter.ToUInt32(System.Text.Encoding.ASCII.GetBytes("DATA"), 0));
        Assert.NotNull(chunk2);
        Assert.Equal(6, chunk2.Data.Length);
    }
}

// <copyright file="ExtendedAssemblerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using System.Runtime.InteropServices;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.SpirV;
using Xunit;

public class ExtendedAssemblerTests
{
    [Fact]
    public void ShaderBytecodeBuilder_ImplementsIShaderBytecodeBuilder_PtxAndSpirvAndDxilQuarantine()
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
        Assert.True(SpirvValidator.Validate(spirv.Span, out string? err), err);

        // Direct DXIL is quarantined with NotSupportedException
        var dxilEx = Assert.Throws<NotSupportedException>(() => builder.BuildDxil(ShaderStage.Compute, instructions));
        Assert.Contains("external DXC/dxil signing", dxilEx.Message);
    }

    [Fact]
    public void SpirvBinaryEmitter_VertexAndPixelStages_OmitComputeOnlyBuiltInsAndLocalSize()
    {
        var emitter = new SpirvBinaryEmitter();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var vertBytes = emitter.EmitModule(ShaderStage.Vertex, instructions);
        Assert.True(SpirvValidator.Validate(vertBytes.Span, out string? vertErr), vertErr);
        AssertDoesNotContainComputeBuiltInOrLocalSize(vertBytes.Span);

        var pixelBytes = emitter.EmitModule(ShaderStage.Pixel, instructions);
        Assert.True(SpirvValidator.Validate(pixelBytes.Span, out string? pixErr), pixErr);
        AssertDoesNotContainComputeBuiltInOrLocalSize(pixelBytes.Span);

        // Compute-only opcodes on Vertex/Pixel must throw NotSupportedException
        var computeOnly = new Instruction[] { new((int)ComputeOpcode.GlobalId, 0, 0, 1) };
        Assert.Throws<NotSupportedException>(() => emitter.EmitModule(ShaderStage.Vertex, computeOnly));
        Assert.Throws<NotSupportedException>(() => emitter.EmitModule(ShaderStage.Pixel, computeOnly));
    }

    private static void AssertDoesNotContainComputeBuiltInOrLocalSize(ReadOnlySpan<byte> spirvBytes)
    {
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(spirvBytes);
        int idx = 5;
        while (idx < words.Length)
        {
            ushort opcode = (ushort)(words[idx] & 0xFFFF);
            ushort count = (ushort)(words[idx] >> 16);
            if (opcode == SpirvOpcode.OpDecorate && count >= 4 &&
                words[idx + 2] == SpirvOpcode.DecorationBuiltIn)
            {
                Assert.NotEqual(SpirvOpcode.BuiltInGlobalInvocationId, words[idx + 3]);
                Assert.NotEqual(SpirvOpcode.BuiltInLocalInvocationId, words[idx + 3]);
            }

            if (opcode == SpirvOpcode.OpExecutionMode && count >= 3)
            {
                Assert.NotEqual(SpirvOpcode.ExecutionModeLocalSize, words[idx + 2]);
            }

            idx += count;
        }
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

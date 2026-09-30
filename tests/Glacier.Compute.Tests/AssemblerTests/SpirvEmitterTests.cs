// <copyright file="SpirvEmitterTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.SpirV;
using Xunit;

public class SpirvEmitterTests
{
    [Fact]
    public void SpirvBinaryEmitter_EmitsValidHeader()
    {
        var emitter = new SpirvBinaryEmitter();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var bytes = emitter.EmitModule(ShaderStage.Compute, instructions);

        Assert.True(bytes.Length >= 20);
        Assert.Equal(0, bytes.Length % 4);

        ReadOnlySpan<byte> span = bytes.Span;
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(span[0..4]);
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(span[4..8]);
        uint generator = BinaryPrimitives.ReadUInt32LittleEndian(span[8..12]);
        uint bound = BinaryPrimitives.ReadUInt32LittleEndian(span[12..16]);
        uint schema = BinaryPrimitives.ReadUInt32LittleEndian(span[16..20]);

        Assert.Equal(SpirvOpcode.MagicNumber, magic);
        Assert.Equal(SpirvOpcode.Version16, version);
        Assert.Equal(SpirvOpcode.GeneratorId, generator);
        Assert.True(bound > 0);
        Assert.Equal(0u, schema);
    }

    [Fact]
    public void SpirvValidator_PassesGeneratedBytecode()
    {
        var emitter = new SpirvBinaryEmitter();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Nop, 0, 0, 0),
            new((int)ComputeOpcode.Add, 0, 1, 2),
            new((int)ComputeOpcode.Sub, 2, 3, 4),
            new((int)ComputeOpcode.Mul, 4, 5, 6),
            new((int)ComputeOpcode.Div, 6, 7, 8),
            new((int)ComputeOpcode.Neg, 8, 0, 9),
            new((int)ComputeOpcode.BarrierSync, 0, 0, 0),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var memory = emitter.EmitModule(ShaderStage.Compute, instructions, localSizeX: 128);

        bool isValid = SpirvValidator.Validate(memory.Span, out string? error);
        Assert.True(isValid, $"SPIR-V validation failed: {error}");
        Assert.Null(error);
    }

    [Fact]
    public void SpirvBinaryEmitter_ZeroAllocationEmitWords_Succeeds()
    {
        var emitter = new SpirvBinaryEmitter();
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        Span<uint> wordBuffer = stackalloc uint[1024];
        int wordsWritten = emitter.EmitWords(ShaderStage.Compute, instructions, wordBuffer);

        Assert.True(wordsWritten > 5);
        Assert.Equal(SpirvOpcode.MagicNumber, wordBuffer[0]);
        Assert.Equal(SpirvOpcode.Version16, wordBuffer[1]);
    }

    [Fact]
    public void SpirvValidator_RejectsInvalidBinary()
    {
        // Truncated buffer
        byte[] truncated = new byte[12];
        Assert.False(SpirvValidator.Validate(truncated, out string? err1));
        Assert.NotNull(err1);

        // Invalid magic
        byte[] badMagic = new byte[24];
        BinaryPrimitives.WriteUInt32LittleEndian(badMagic.AsSpan(0, 4), 0x12345678);
        BinaryPrimitives.WriteUInt32LittleEndian(badMagic.AsSpan(4, 4), 0x00010600);
        BinaryPrimitives.WriteUInt32LittleEndian(badMagic.AsSpan(12, 4), 10);
        Assert.False(SpirvValidator.Validate(badMagic, out string? err2));
        Assert.NotNull(err2);

        // Bound is zero
        byte[] badBound = new byte[24];
        BinaryPrimitives.WriteUInt32LittleEndian(badBound.AsSpan(0, 4), SpirvOpcode.MagicNumber);
        BinaryPrimitives.WriteUInt32LittleEndian(badBound.AsSpan(4, 4), 0x00010600);
        BinaryPrimitives.WriteUInt32LittleEndian(badBound.AsSpan(12, 4), 0); // bound = 0
        Assert.False(SpirvValidator.Validate(badBound, out string? err3));
        Assert.NotNull(err3);
    }
}

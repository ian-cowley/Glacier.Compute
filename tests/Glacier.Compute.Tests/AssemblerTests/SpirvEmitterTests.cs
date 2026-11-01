// <copyright file="SpirvEmitterTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
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
    public void SpirvBinaryEmitter_EmitsStorageBuffers_GlslStd450_AndAllComputeOpcodes()
    {
        var emitter = new SpirvBinaryEmitter();
        int constBits = BitConverter.SingleToInt32Bits(2.5f);

        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.GlobalId, 0, 0, 0),
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.LoadGlobal, 0, 0, 2),
            new((int)ComputeOpcode.LoadGlobal, 1, 0, 3),
            new((int)ComputeOpcode.LoadConstant, constBits, 0, 4),
            new((int)ComputeOpcode.Add, 2, 3, 5),
            new((int)ComputeOpcode.Sub, 5, 4, 6),
            new((int)ComputeOpcode.Mul, 6, 4, 7),
            new((int)ComputeOpcode.Div, 7, 4, 8),
            new((int)ComputeOpcode.Fma, 2, 3, 8),
            new((int)ComputeOpcode.Neg, 8, 0, 9),
            new((int)ComputeOpcode.Abs, 9, 0, 10),
            new((int)ComputeOpcode.Min, 10, 4, 11),
            new((int)ComputeOpcode.Max, 11, 4, 12),
            new((int)ComputeOpcode.Sqrt, 12, 0, 13),
            new((int)ComputeOpcode.Rsqrt, 13, 0, 14),
            new((int)ComputeOpcode.Exp, 14, 0, 15),
            new((int)ComputeOpcode.Log, 15, 0, 16),
            new((int)ComputeOpcode.Tanh, 16, 0, 17),
            new((int)ComputeOpcode.StoreShared, 17, 1, 0),
            new((int)ComputeOpcode.BarrierSync, 0, 0, 0),
            new((int)ComputeOpcode.LoadShared, 0, 1, 18),
            new((int)ComputeOpcode.ShflSyncBfly, 18, 16, 19),
            new((int)ComputeOpcode.StoreGlobal, 19, 0, 0),
            new((int)ComputeOpcode.AtomicAdd, 19, 0, 20),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var spirv = emitter.EmitModule(ShaderStage.Compute, instructions, localSizeX: 128);
        Assert.True(SpirvValidator.Validate(spirv.Span, out string? error), $"Validation failed: {error}");

        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(spirv.Span);
        var seenOpcodes = new HashSet<ushort>();
        var seenGlslOps = new HashSet<uint>();
        var seenBindings = new HashSet<uint>();
        int storageBufferVarCount = 0;
        int pushConstVarCount = 0;
        int workgroupVarCount = 0;
        int opReturnCount = 0;
        int opNopCount = 0;

        int i = 5;
        while (i < words.Length)
        {
            ushort op = (ushort)(words[i] & 0xFFFF);
            ushort wc = (ushort)(words[i] >> 16);
            seenOpcodes.Add(op);

            if (op == SpirvOpcode.OpExtInst && wc >= 5)
            {
                seenGlslOps.Add(words[i + 4]);
            }
            else if (op == SpirvOpcode.OpDecorate && wc >= 4 && words[i + 2] == SpirvOpcode.DecorationBinding)
            {
                seenBindings.Add(words[i + 3]);
            }
            else if (op == SpirvOpcode.OpVariable && wc >= 4)
            {
                uint storageClass = words[i + 3];
                if (storageClass == SpirvOpcode.StorageClassStorageBuffer) storageBufferVarCount++;
                else if (storageClass == SpirvOpcode.StorageClassPushConstant) pushConstVarCount++;
                else if (storageClass == SpirvOpcode.StorageClassWorkgroup) workgroupVarCount++;
            }
            else if (op == SpirvOpcode.OpReturn)
            {
                opReturnCount++;
            }
            else if (op == SpirvOpcode.OpNop)
            {
                opNopCount++;
            }

            i += wc;
        }

        // Verify zero silent OpNop emissions when ComputeOpcode.Nop was not requested
        Assert.Equal(0, opNopCount);

        // Verify strictly single OpReturn
        Assert.Equal(1, opReturnCount);

        // Verify StorageBuffer (bindings 0, 1, 2), PushConstant (param_n), and Workgroup shared memory
        Assert.Equal(3, storageBufferVarCount);
        Assert.Contains(0u, seenBindings);
        Assert.Contains(1u, seenBindings);
        Assert.Contains(2u, seenBindings);
        Assert.Equal(1, pushConstVarCount);
        Assert.Equal(1, workgroupVarCount);

        // Verify GLSL.std.450 import & all 9 extended math instructions
        Assert.Contains(SpirvOpcode.OpExtInstImport, seenOpcodes);
        Assert.Contains(SpirvOpcode.GlslStd450Fma, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450FAbs, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450FMin, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450FMax, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450Sqrt, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450InverseSqrt, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450Exp, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450Log, seenGlslOps);
        Assert.Contains(SpirvOpcode.GlslStd450Tanh, seenGlslOps);

        // Verify memory & atomic & subgroup ops
        Assert.Contains(SpirvOpcode.OpAccessChain, seenOpcodes);
        Assert.Contains(SpirvOpcode.OpAtomicFAddEXT, seenOpcodes);
        Assert.Contains(SpirvOpcode.OpGroupNonUniformShuffleXor, seenOpcodes);
        Assert.Contains(SpirvOpcode.OpControlBarrier, seenOpcodes);
    }

    [Fact]
    public void SpirvBinaryEmitter_SingleOpReturn_WithOrWithoutExplicitReturn()
    {
        var emitter = new SpirvBinaryEmitter();

        // Case 1: Ends with explicit Return
        var withReturn = emitter.EmitModule(ShaderStage.Compute, new Instruction[]
        {
            new((int)ComputeOpcode.Add, 0, 1, 2),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        });
        Assert.True(SpirvValidator.Validate(withReturn.Span, out string? err1), err1);
        Assert.Equal(1, CountOpcode(withReturn.Span, SpirvOpcode.OpReturn));

        // Case 2: No explicit Return -> emitter appends single OpReturn
        var withoutReturn = emitter.EmitModule(ShaderStage.Compute, new Instruction[]
        {
            new((int)ComputeOpcode.Add, 0, 1, 2)
        });
        Assert.True(SpirvValidator.Validate(withoutReturn.Span, out string? err2), err2);
        Assert.Equal(1, CountOpcode(withoutReturn.Span, SpirvOpcode.OpReturn));

        // Case 3: Multiple Return instructions in IR -> only one OpReturn emitted in block
        var multipleReturns = emitter.EmitModule(ShaderStage.Compute, new Instruction[]
        {
            new((int)ComputeOpcode.Return, 0, 0, 0),
            new((int)ComputeOpcode.Add, 0, 1, 2),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        });
        Assert.True(SpirvValidator.Validate(multipleReturns.Span, out string? err3), err3);
        Assert.Equal(1, CountOpcode(multipleReturns.Span, SpirvOpcode.OpReturn));
    }

    [Fact]
    public void SpirvBinaryEmitter_ThrowsNotSupportedException_OnInvalidOpcodes()
    {
        var emitter = new SpirvBinaryEmitter();

        Assert.Throws<NotSupportedException>(() =>
            emitter.EmitModule(ShaderStage.Compute, new Instruction[] { new(999, 0, 0, 0) }));

        Assert.Throws<NotSupportedException>(() =>
            emitter.EmitModule(ShaderStage.Compute, new Instruction[] { new(-1, 0, 0, 0) }));

        // Invalid opcode even after Return must still be caught during IR validation
        Assert.Throws<NotSupportedException>(() =>
            emitter.EmitModule(ShaderStage.Compute, new Instruction[]
            {
                new((int)ComputeOpcode.Return, 0, 0, 0),
                new(777, 0, 0, 0)
            }));
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
    public void SpirvValidator_RejectsInvalidBinary_DoubleReturn_AndOutOfBoundsIds()
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
        BinaryPrimitives.WriteUInt32LittleEndian(badBound.AsSpan(12, 4), 0);
        Assert.False(SpirvValidator.Validate(badBound, out string? err3));
        Assert.NotNull(err3);

        // Duplicate OpReturn in same basic block must fail validation
        var emitter = new SpirvBinaryEmitter();
        byte[] validModule = emitter.EmitModule(ShaderStage.Compute, new Instruction[] { new((int)ComputeOpcode.Return, 0, 0, 0) }).ToArray();
        Assert.True(SpirvValidator.Validate(validModule, out _));

        // Insert a second OpReturn right before OpFunctionEnd (last word)
        byte[] doubleReturn = new byte[validModule.Length + 4];
        validModule.AsSpan(0, validModule.Length - 4).CopyTo(doubleReturn);
        BinaryPrimitives.WriteUInt32LittleEndian(doubleReturn.AsSpan(validModule.Length - 4, 4), (1u << 16) | SpirvOpcode.OpReturn);
        BinaryPrimitives.WriteUInt32LittleEndian(doubleReturn.AsSpan(validModule.Length, 4), (1u << 16) | SpirvOpcode.OpFunctionEnd);
        Assert.False(SpirvValidator.Validate(doubleReturn, out string? dupErr));
        Assert.Contains("Duplicate block terminator", dupErr);

        // Corrupt header bound to 2 so existing IDs exceed bound -> must fail id < bound check
        byte[] outOfBoundsId = (byte[])validModule.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(outOfBoundsId.AsSpan(12, 4), 2u);
        Assert.False(SpirvValidator.Validate(outOfBoundsId, out string? boundErr));
        Assert.Contains("Invalid SPIR-V ID", boundErr);
    }

    private static int CountOpcode(ReadOnlySpan<byte> spirvBytes, ushort targetOpcode)
    {
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(spirvBytes);
        int count = 0;
        int i = 5;
        while (i < words.Length)
        {
            ushort op = (ushort)(words[i] & 0xFFFF);
            ushort wc = (ushort)(words[i] >> 16);
            if (op == targetOpcode)
            {
                count++;
            }
            i += wc;
        }
        return count;
    }
}

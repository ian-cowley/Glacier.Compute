// <copyright file="DxbcContainerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using System.Buffers.Binary;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Xunit;

public class DxbcContainerTests
{
    [Fact]
    public void DxbcContainerBuilder_BuildsValidComputeContainer()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var container = DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions);

        Assert.True(container.Length > 32);

        // Parse container with reader
        var reader = DxbcContainerReader.Parse(container.Span);
        Assert.Equal(1u, reader.Version);
        Assert.Equal((uint)container.Length, reader.TotalSize);
        Assert.Equal(2, reader.Chunks.Count);

        // Verify DXIL chunk
        var dxilChunk = reader.FindChunk(DxbcChunk.FourCcDxil);
        Assert.NotNull(dxilChunk);
        Assert.Equal("DXIL", dxilChunk.FourCcString);
        Assert.True(dxilChunk.Data.Length >= 16);

        // Verify DXIL header inside chunk data
        ReadOnlySpan<byte> dxilSpan = dxilChunk.Data;
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(dxilSpan[0..2]);
        ushort shaderModel = BinaryPrimitives.ReadUInt16LittleEndian(dxilSpan[2..4]);
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[4..8]);
        uint bitcodeOffset = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[8..12]);
        uint bitcodeSize = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[12..16]);

        Assert.Equal(0x0100, version);
        Assert.Equal(DxbcChunk.FourCcDxil, magic);
        Assert.Equal(16u, bitcodeOffset);
        Assert.True(bitcodeSize > 0);

        // Verify LLVM bitcode magic at offset 16
        uint llvmMagic = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[16..20]);
        Assert.Equal(DxilChunkEmitter.LlvmBitcodeMagic, llvmMagic);

        // Verify PSV0 chunk
        var psv0Chunk = reader.FindChunk(DxbcChunk.FourCcPsv0);
        Assert.NotNull(psv0Chunk);
        Assert.Equal("PSV0", psv0Chunk.FourCcString);
    }

    [Fact]
    public void DxbcContainerReader_RejectsInvalidContainers()
    {
        // Truncated
        byte[] truncated = new byte[16];
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(truncated));

        // Bad magic
        byte[] badMagic = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(badMagic.AsSpan(0, 4), 0x11223344);
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(badMagic));

        // Size mismatch
        byte[] sizeMismatch = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(0, 4), DxbcContainerBuilder.MagicFourCc);
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(20, 4), 1); // version
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(24, 4), 100); // reports 100, actual 32
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(sizeMismatch));
    }

    [Fact]
    public void DxbcContainerBuilder_ComputesConsistentChecksum()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3)
        };

        var container1 = DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions);
        var container2 = DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions);

        var reader1 = DxbcContainerReader.Parse(container1.Span);
        var reader2 = DxbcContainerReader.Parse(container2.Span);

        Assert.True(reader1.Checksum.Span.SequenceEqual(reader2.Checksum.Span));
    }
}

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
    public void DxilDirectBitcodeEmission_ThrowsNotSupportedException()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        var ex1 = Assert.Throws<NotSupportedException>(() => DxilChunkEmitter.CreateDxilChunk(ShaderStage.Compute, instructions));
        Assert.Contains("Direct LLVM 3.7 DXIL bitcode emission requires external DXC/dxil signing", ex1.Message);

        var ex2 = Assert.Throws<NotSupportedException>(() => DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, instructions));
        Assert.Contains("Direct LLVM 3.7 DXIL bitcode emission requires external DXC/dxil signing", ex2.Message);
    }

    [Fact]
    public void DxbcContainerBuilder_PackagesPrecompiledBitcodeBlob()
    {
        byte[] precompiledBitcode = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(precompiledBitcode.AsSpan(0, 4), DxilChunkEmitter.LlvmBitcodeMagic);
        precompiledBitcode[4] = 0x21;
        precompiledBitcode[5] = 0x0C;

        var container = DxbcContainerBuilder.PackagePrecompiledShader(ShaderStage.Compute, precompiledBitcode);
        Assert.True(container.Length > 32);

        var reader = DxbcContainerReader.Parse(container.Span);
        Assert.Equal(1u, reader.Version);
        Assert.Equal((uint)container.Length, reader.TotalSize);
        Assert.Equal(2, reader.Chunks.Count);

        var dxilChunk = reader.FindChunk(DxbcChunk.FourCcDxil);
        Assert.NotNull(dxilChunk);
        Assert.Equal("DXIL", dxilChunk.FourCcString);
        Assert.True(dxilChunk.Data.Length >= 16 + precompiledBitcode.Length);

        ReadOnlySpan<byte> dxilSpan = dxilChunk.Data;
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(dxilSpan[0..2]);
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[4..8]);
        uint bitcodeOffset = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[8..12]);
        uint bitcodeSize = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[12..16]);

        Assert.Equal(0x0100, version);
        Assert.Equal(DxbcChunk.FourCcDxil, magic);
        Assert.Equal(16u, bitcodeOffset);
        Assert.Equal((uint)precompiledBitcode.Length, bitcodeSize);

        uint llvmMagic = BinaryPrimitives.ReadUInt32LittleEndian(dxilSpan[16..20]);
        Assert.Equal(DxilChunkEmitter.LlvmBitcodeMagic, llvmMagic);

        var psv0Chunk = reader.FindChunk(DxbcChunk.FourCcPsv0);
        Assert.NotNull(psv0Chunk);
        Assert.Equal("PSV0", psv0Chunk.FourCcString);
    }

    [Fact]
    public void DxilChunkEmitter_RejectsInvalidPrecompiledMagic()
    {
        byte[] badBlob = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Assert.Throws<ArgumentException>(() => DxilChunkEmitter.CreateDxilChunkFromBitcode(ShaderStage.Compute, badBlob));
        Assert.Throws<ArgumentException>(() => DxilChunkEmitter.CreateDxilChunkFromBitcode(ShaderStage.Compute, new byte[] { 0x42, 0x43 }));
    }

    [Fact]
    public void DxbcContainerReader_RejectsInvalidContainers()
    {
        byte[] truncated = new byte[16];
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(truncated));

        byte[] badMagic = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(badMagic.AsSpan(0, 4), 0x11223344);
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(badMagic));

        byte[] sizeMismatch = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(0, 4), DxbcContainerBuilder.MagicFourCc);
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(20, 4), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(sizeMismatch.AsSpan(24, 4), 100);
        Assert.Throws<FormatException>(() => DxbcContainerReader.Parse(sizeMismatch));
    }

    [Fact]
    public void DxbcContainerBuilder_ComputesConsistentChecksum()
    {
        byte[] precompiledBitcode = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(precompiledBitcode.AsSpan(0, 4), DxilChunkEmitter.LlvmBitcodeMagic);

        var container1 = DxbcContainerBuilder.PackagePrecompiledShader(ShaderStage.Compute, precompiledBitcode);
        var container2 = DxbcContainerBuilder.PackagePrecompiledShader(ShaderStage.Compute, precompiledBitcode);

        var reader1 = DxbcContainerReader.Parse(container1.Span);
        var reader2 = DxbcContainerReader.Parse(container2.Span);

        Assert.True(reader1.Checksum.Span.SequenceEqual(reader2.Checksum.Span));
    }
}

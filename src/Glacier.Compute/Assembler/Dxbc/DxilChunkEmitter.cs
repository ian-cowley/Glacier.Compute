// <copyright file="DxilChunkEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Dxbc;

using System;
using System.Buffers.Binary;
using System.IO;

/// <summary>
/// Packages precompiled LLVM 3.7 DXIL bitcode and PSV0 metadata chunks for Direct3D 12 DXBC containers.
/// Direct LLVM 3.7 bitcode synthesis from <see cref="Instruction"/> streams is intentionally quarantined
/// because retail D3D12 drivers require DXC compilation and <c>dxil.dll</c> validator signing.
/// </summary>
public static class DxilChunkEmitter
{
    /// <summary>LLVM 3.7 Bitcode container magic number (0xDEC04342 / 'BC\xC0\xDE').</summary>
    public const uint LlvmBitcodeMagic = 0xDEC04342;

    /// <summary>LLVM Bitcode wrapper magic number (0x0B17C0DE).</summary>
    public const uint LlvmBitcodeWrapperMagic = 0x0B17C0DE;

    /// <summary>Compute shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindCompute = 5;

    /// <summary>Vertex shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindVertex = 1;

    /// <summary>Pixel / Fragment shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindPixel = 0;

    internal const string QuarantineMessage =
        "Direct LLVM 3.7 DXIL bitcode emission requires external DXC/dxil signing; use DXBC container packaging with precompiled shader blobs or SPIR-V/PTX targets.";

    /// <summary>
    /// Quarantined entry point: throws <see cref="NotSupportedException"/> because retail Direct3D 12 drivers
    /// reject unsigned synthetic LLVM 3.7 bitcode streams without external DXC compilation and <c>dxil.dll</c> signing.
    /// Use <see cref="CreateDxilChunkFromBitcode"/> with a precompiled DXC bitcode payload instead.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown to prevent silent emission of driver-rejected DXIL bitcode.</exception>
    public static DxbcChunk CreateDxilChunk(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        _ = stage;
        _ = instructions;
        throw new NotSupportedException(QuarantineMessage);
    }

    /// <summary>
    /// Wraps a precompiled LLVM 3.7 DXIL bitcode payload in a standard 16-byte DXIL program header chunk.
    /// </summary>
    /// <param name="stage">Target shader stage.</param>
    /// <param name="precompiledBitcode">Precompiled LLVM 3.7 bitcode bytes starting with magic <c>0xDEC04342</c> or <c>0x0B17C0DE</c>.</param>
    /// <param name="majorVersion">DXIL shader model major version (default 6).</param>
    /// <param name="minorVersion">DXIL shader model minor version (default 0).</param>
    /// <returns>A formatted <c>DXIL</c> <see cref="DxbcChunk"/>.</returns>
    public static DxbcChunk CreateDxilChunkFromBitcode(
        ShaderStage stage,
        ReadOnlySpan<byte> precompiledBitcode,
        byte majorVersion = 6,
        byte minorVersion = 0)
    {
        if (precompiledBitcode.Length < 4)
        {
            throw new ArgumentException("Precompiled DXIL bitcode must be at least 4 bytes and start with the LLVM bitcode magic header.", nameof(precompiledBitcode));
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(precompiledBitcode[..4]);
        if (magic != LlvmBitcodeMagic && magic != LlvmBitcodeWrapperMagic)
        {
            throw new ArgumentException($"Invalid LLVM bitcode magic 0x{magic:X8}; expected 0x{LlvmBitcodeMagic:X8} ('BC\\xC0\\xDE') or 0x{LlvmBitcodeWrapperMagic:X8}.", nameof(precompiledBitcode));
        }

        int alignedBitcodeLen = (precompiledBitcode.Length + 3) & ~3;
        byte[] payload = new byte[16 + alignedBitcodeLen];
        Span<byte> span = payload;

        ushort kind = stage switch
        {
            ShaderStage.Vertex => DxilShaderKindVertex,
            ShaderStage.Pixel => DxilShaderKindPixel,
            _ => DxilShaderKindCompute
        };

        ushort shaderModel = (ushort)((kind << 8) | ((majorVersion & 0x0F) << 4) | (minorVersion & 0x0F));

        // 16-byte DXIL program header
        BinaryPrimitives.WriteUInt16LittleEndian(span[0..2], 0x0100);
        BinaryPrimitives.WriteUInt16LittleEndian(span[2..4], shaderModel);
        BinaryPrimitives.WriteUInt32LittleEndian(span[4..8], DxbcChunk.FourCcDxil);
        BinaryPrimitives.WriteUInt32LittleEndian(span[8..12], 16u);
        BinaryPrimitives.WriteUInt32LittleEndian(span[12..16], (uint)precompiledBitcode.Length);

        precompiledBitcode.CopyTo(span[16..]);
        return new DxbcChunk(DxbcChunk.FourCcDxil, payload);
    }

    /// <summary>
    /// Creates a PSV0 (Pipeline State Validation) chunk for compute/graphics container packaging.
    /// </summary>
    public static DxbcChunk CreatePsv0Chunk(ShaderStage stage, uint threadsX = 64, uint threadsY = 1, uint threadsZ = 1)
    {
        using var ms = new MemoryStream(32);
        using var writer = new BinaryWriter(ms);

        writer.Write((uint)2); // PSV version
        writer.Write((uint)(stage == ShaderStage.Compute ? 5 : (stage == ShaderStage.Vertex ? 1 : 0)));
        writer.Write(threadsX);
        writer.Write(threadsY);
        writer.Write(threadsZ);

        writer.Write((uint)0); // Input signature count
        writer.Write((uint)0); // Output signature count
        writer.Write((uint)3); // Resource binding count

        return new DxbcChunk(DxbcChunk.FourCcPsv0, ms.ToArray());
    }
}

// <copyright file="DxbcContainerBuilder.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Dxbc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;

/// <summary>
/// Pure C# DirectX Container (DXBC) packager for precompiled shader chunks and container inspection.
/// Note: Retail D3D12 execution of SM 6.0+ DXIL requires external <c>dxil.dll</c> validator signing unless
/// D3D12 experimental shader models Developer Mode is active; this builder writes an unsigned MD5 digest
/// over container payload bytes (<c>offset 20..end</c>) for structural integrity checks.
/// </summary>
public sealed class DxbcContainerBuilder
{
    /// <summary>DXBC container magic number ('DXBC' = 0x43425844).</summary>
    public const uint MagicFourCc = 0x43425844;

    private readonly List<DxbcChunk> _chunks = new();

    /// <summary>
    /// Adds a chunk to the container.
    /// </summary>
    public DxbcContainerBuilder AddChunk(DxbcChunk chunk)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        _chunks.Add(chunk);
        return this;
    }

    /// <summary>
    /// Builds the complete DXBC container byte array.
    /// </summary>
    public byte[] Build()
    {
        if (_chunks.Count == 0)
        {
            throw new InvalidOperationException("Cannot build DXBC container with zero chunks.");
        }

        int headerSize = 32 + (_chunks.Count * 4);

        int totalSize = headerSize;
        for (int i = 0; i < _chunks.Count; i++)
        {
            totalSize += 8 + _chunks[i].Data.Length;
        }

        byte[] buffer = new byte[totalSize];
        using var ms = new MemoryStream(buffer);
        using var writer = new BinaryWriter(ms);

        // 1. Magic 'DXBC'
        writer.Write(MagicFourCc);

        // 2. Placeholder for 16-byte digest (offsets 4..19)
        writer.Write(new byte[16]);

        // 3. Version (1)
        writer.Write((uint)1);

        // 4. Total Size
        writer.Write((uint)totalSize);

        // 5. Chunk Count
        writer.Write((uint)_chunks.Count);

        // 6. Write chunk offsets
        int currentChunkOffset = headerSize;
        for (int i = 0; i < _chunks.Count; i++)
        {
            writer.Write((uint)currentChunkOffset);
            currentChunkOffset += 8 + _chunks[i].Data.Length;
        }

        // 7. Write chunks
        for (int i = 0; i < _chunks.Count; i++)
        {
            var chunk = _chunks[i];
            writer.Write(chunk.FourCc);
            writer.Write((uint)chunk.Data.Length);
            writer.Write(chunk.Data);
        }

        // 8. Compute unsigned integrity digest over container body (byte 20 onwards)
        MD5.HashData(buffer.AsSpan(20), buffer.AsSpan(4, 16));

        return buffer;
    }

    /// <summary>
    /// Quarantined entry point: throws <see cref="NotSupportedException"/> because direct LLVM 3.7 DXIL synthesis
    /// from <see cref="Instruction"/> streams requires external DXC compilation and <c>dxil.dll</c> signing.
    /// Use <see cref="PackagePrecompiledShader"/> to package precompiled LLVM bitcode blobs.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    public static ReadOnlyMemory<byte> BuildComputeShader(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        var dxilChunk = DxilChunkEmitter.CreateDxilChunk(stage, instructions);
        var psv0Chunk = DxilChunkEmitter.CreatePsv0Chunk(stage, threadsX: 64, threadsY: 1, threadsZ: 1);

        var builder = new DxbcContainerBuilder();
        builder.AddChunk(dxilChunk);
        builder.AddChunk(psv0Chunk);
        return builder.Build();
    }

    /// <summary>
    /// Packages a precompiled LLVM 3.7 DXIL bitcode blob and PSV0 metadata chunk into a DXBC container.
    /// </summary>
    public static ReadOnlyMemory<byte> PackagePrecompiledShader(
        ShaderStage stage,
        ReadOnlySpan<byte> precompiledBitcode,
        uint threadsX = 64,
        uint threadsY = 1,
        uint threadsZ = 1)
    {
        var builder = new DxbcContainerBuilder();
        builder.AddChunk(DxilChunkEmitter.CreateDxilChunkFromBitcode(stage, precompiledBitcode));
        builder.AddChunk(DxilChunkEmitter.CreatePsv0Chunk(stage, threadsX, threadsY, threadsZ));
        return builder.Build();
    }
}

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
/// Direct pure C# DirectX Container (DXBC) builder for Direct3D 12 compute pipelines.
/// Assembles container chunks and computes container checksums without external toolchains.
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

        // Header size:
        // 4 bytes: Magic 'DXBC'
        // 16 bytes: MD5 / Hash digest
        // 4 bytes: Version (1)
        // 4 bytes: TotalSize
        // 4 bytes: ChunkCount
        // ChunkCount * 4 bytes: Chunk offsets
        int headerSize = 32 + (_chunks.Count * 4);

        int totalSize = headerSize;
        for (int i = 0; i < _chunks.Count; i++)
        {
            // Each chunk has 4 bytes FourCC + 4 bytes Size + Data
            totalSize += 8 + _chunks[i].Data.Length;
        }

        byte[] buffer = new byte[totalSize];
        using var ms = new MemoryStream(buffer);
        using var writer = new BinaryWriter(ms);

        // 1. Magic 'DXBC'
        writer.Write(MagicFourCc);

        // 2. Placeholder for 16-byte checksum (offsets 4..19)
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

        // 8. Compute MD5 checksum over container (excluding checksum field itself)
        byte[] hash = MD5.HashData(buffer.AsSpan(20)); // Hash from byte 20 onwards
        Array.Copy(hash, 0, buffer, 4, 16);

        return buffer;
    }

    /// <summary>
    /// Helper to assemble a complete compute shader DXBC container with DXIL and PSV0 chunks.
    /// </summary>
    public static ReadOnlyMemory<byte> BuildComputeShader(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        var builder = new DxbcContainerBuilder();

        // 1. DXIL chunk
        var dxilChunk = DxilChunkEmitter.CreateDxilChunk(stage, instructions);
        builder.AddChunk(dxilChunk);

        // 2. PSV0 chunk
        var psv0Chunk = DxilChunkEmitter.CreatePsv0Chunk(stage, threadsX: 64, threadsY: 1, threadsZ: 1);
        builder.AddChunk(psv0Chunk);

        return builder.Build();
    }
}

// <copyright file="DxbcContainerReader.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Dxbc;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;

/// <summary>
/// Parser and validator for DirectX Container (DXBC) binary blobs.
/// </summary>
public sealed class DxbcContainerReader
{
    /// <summary>Gets the list of parsed chunks in the container.</summary>
    public IReadOnlyList<DxbcChunk> Chunks { get; }

    /// <summary>Gets the 16-byte container checksum.</summary>
    public ReadOnlyMemory<byte> Checksum { get; }

    /// <summary>Gets the total size of the container in bytes.</summary>
    public uint TotalSize { get; }

    /// <summary>Gets the container version.</summary>
    public uint Version { get; }

    private DxbcContainerReader(List<DxbcChunk> chunks, byte[] checksum, uint totalSize, uint version)
    {
        Chunks = chunks;
        Checksum = checksum;
        TotalSize = totalSize;
        Version = version;
    }

    /// <summary>
    /// Parses a DXBC container from a byte span.
    /// </summary>
    public static DxbcContainerReader Parse(ReadOnlySpan<byte> data)
    {
        if (data.Length < 32)
        {
            throw new FormatException($"Invalid DXBC container: length {data.Length} is less than minimum header size of 32 bytes.");
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data[0..4]);
        if (magic != DxbcContainerBuilder.MagicFourCc)
        {
            throw new FormatException($"Invalid DXBC magic: 0x{magic:X8}, expected 0x{DxbcContainerBuilder.MagicFourCc:X8} ('DXBC').");
        }

        byte[] checksum = data.Slice(4, 16).ToArray();
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(data[20..24]);
        uint totalSize = BinaryPrimitives.ReadUInt32LittleEndian(data[24..28]);
        uint chunkCount = BinaryPrimitives.ReadUInt32LittleEndian(data[28..32]);

        if (totalSize != (uint)data.Length)
        {
            throw new FormatException($"DXBC container size mismatch: header reports {totalSize} bytes, actual buffer is {data.Length} bytes.");
        }

        int offsetTableEnd = 32 + (int)(chunkCount * 4);
        if (data.Length < offsetTableEnd)
        {
            throw new FormatException("DXBC container offset table exceeds buffer bounds.");
        }

        var chunks = new List<DxbcChunk>((int)chunkCount);
        for (int i = 0; i < chunkCount; i++)
        {
            int offsetPos = 32 + (i * 4);
            uint chunkOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offsetPos, 4));

            if (chunkOffset + 8 > data.Length)
            {
                throw new FormatException($"DXBC chunk offset {chunkOffset} exceeds container length.");
            }

            uint fourCc = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice((int)chunkOffset, 4));
            uint chunkSize = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice((int)chunkOffset + 4, 4));

            if (chunkOffset + 8 + chunkSize > data.Length)
            {
                throw new FormatException($"DXBC chunk payload extends past container length: offset={chunkOffset}, size={chunkSize}.");
            }

            byte[] chunkData = data.Slice((int)chunkOffset + 8, (int)chunkSize).ToArray();
            chunks.Add(new DxbcChunk(fourCc, chunkData));
        }

        return new DxbcContainerReader(chunks, checksum, totalSize, version);
    }

    /// <summary>
    /// Finds the first chunk matching the specified FourCC code, or null if not found.
    /// </summary>
    public DxbcChunk? FindChunk(uint fourCc)
    {
        for (int i = 0; i < Chunks.Count; i++)
        {
            if (Chunks[i].FourCc == fourCc)
            {
                return Chunks[i];
            }
        }
        return null;
    }
}

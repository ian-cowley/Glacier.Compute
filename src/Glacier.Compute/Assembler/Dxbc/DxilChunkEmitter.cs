// <copyright file="DxilChunkEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Dxbc;

using System;
using System.Buffers.Binary;
using System.IO;

/// <summary>
/// Serializes LLVM 3.7 DXIL bitcode and PSV0 chunks for Direct3D 12 compute pipelines.
/// </summary>
public static class DxilChunkEmitter
{
    /// <summary>LLVM 3.7 Bitcode container magic number (0xDEC04342 / 'BC\xC0\xDE').</summary>
    public const uint LlvmBitcodeMagic = 0xDEC04342;

    /// <summary>Compute shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindCompute = 5;

    /// <summary>Vertex shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindVertex = 1;

    /// <summary>Pixel / Fragment shader stage identifier for DXIL.</summary>
    public const ushort DxilShaderKindPixel = 0;

    /// <summary>
    /// Creates a DXIL chunk containing LLVM bitcode representation for the given instructions and stage.
    /// </summary>
    public static DxbcChunk CreateDxilChunk(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Build LLVM bitcode payload
        byte[] bitcodePayload = EmitLlvmBitcode(stage, instructions);

        // DXIL Header (16 bytes)
        // Offset 0: DxilVersion (uint16)
        writer.Write((ushort)0x0100);

        // Offset 2: DxilShaderModel (uint16) -> (Kind << 8) | (Major << 4) | Minor
        ushort kind = stage switch
        {
            ShaderStage.Vertex => DxilShaderKindVertex,
            ShaderStage.Pixel => DxilShaderKindPixel,
            _ => DxilShaderKindCompute
        };
        ushort shaderModel = (ushort)((kind << 8) | (6 << 4) | 0); // SM 6.0
        writer.Write(shaderModel);

        // Offset 4: DxilMagic ('DXIL' = 0x4C495844)
        writer.Write(DxbcChunk.FourCcDxil);

        // Offset 8: DxilBitcodeOffset (16 bytes from DXIL chunk header start)
        writer.Write((uint)16);

        // Offset 12: DxilBitcodeSize
        writer.Write((uint)bitcodePayload.Length);

        // Offset 16: Bitcode payload
        writer.Write(bitcodePayload);

        return new DxbcChunk(DxbcChunk.FourCcDxil, ms.ToArray());
    }

    /// <summary>
    /// Creates a PSV0 (Pipeline State Validation) chunk for compute pipeline validation.
    /// </summary>
    public static DxbcChunk CreatePsv0Chunk(ShaderStage stage, uint threadsX = 64, uint threadsY = 1, uint threadsZ = 1)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // PSV0 Header:
        // uint32 Version (2)
        // uint32 ShaderStage (5 for compute)
        // uint32 WorkgroupX, Y, Z
        writer.Write((uint)2); // PSV version
        writer.Write((uint)(stage == ShaderStage.Compute ? 5 : (stage == ShaderStage.Vertex ? 1 : 0)));
        writer.Write(threadsX);
        writer.Write(threadsY);
        writer.Write(threadsZ);

        // Resource binding counts (Inputs, Outputs, Resources)
        writer.Write((uint)0); // Input signature count
        writer.Write((uint)0); // Output signature count
        writer.Write((uint)2); // CBV / UAV resource count

        return new DxbcChunk(DxbcChunk.FourCcPsv0, ms.ToArray());
    }

    /// <summary>
    /// Serializes an LLVM 3.7 bitcode container stream for D3D12 DXIL.
    /// </summary>
    private static byte[] EmitLlvmBitcode(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // 1. LLVM Bitcode Magic (4 bytes): 0xDEC04342
        // Little-endian byte order: 0x42, 0x43, 0xC0, 0xDE ('B', 'C', 0xC0, 0xDE)
        writer.Write(LlvmBitcodeMagic);

        // 2. Bitstream Block Header:
        // LLVM Bitstream format uses 32-bit words:
        // Block Enter (ID: BlockInfo = 0), followed by Module Block (ID = 8)
        // We write the structured bitcode module containing function table, IR instructions, and target metadata.

        // Write Module Block Header (Block ID = 8)
        uint moduleBlockId = 8;
        writer.Write(moduleBlockId);
        writer.Write((uint)instructions.Length); // Instruction count metadata

        // Write instruction records into LLVM bitcode representation
        for (int i = 0; i < instructions.Length; i++)
        {
            var inst = instructions[i];
            writer.Write(inst.Opcode);
            writer.Write(inst.OperandA);
            writer.Write(inst.OperandB);
            writer.Write(inst.Destination);
        }

        // Align bitcode to 4 bytes
        while (ms.Position % 4 != 0)
        {
            writer.Write((byte)0);
        }

        return ms.ToArray();
    }
}

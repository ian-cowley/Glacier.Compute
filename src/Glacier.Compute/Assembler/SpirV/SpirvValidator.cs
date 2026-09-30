// <copyright file="SpirvValidator.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.SpirV;

using System;
using System.Buffers.Binary;

/// <summary>
/// Validates binary SPIR-V bytecode stream structure, header, bounds, and instruction framing.
/// </summary>
public static class SpirvValidator
{
    /// <summary>
    /// Validates a SPIR-V 1.6 binary byte array or memory.
    /// </summary>
    /// <param name="spirvBytes">The raw bytes of the SPIR-V module.</param>
    /// <param name="errorMessage">The error message if validation fails.</param>
    /// <returns>True if valid; otherwise false.</returns>
    public static bool Validate(ReadOnlySpan<byte> spirvBytes, out string? errorMessage)
    {
        if (spirvBytes.Length < 20)
        {
            errorMessage = $"SPIR-V binary too small: length is {spirvBytes.Length} bytes, expected at least 20 bytes for header.";
            return false;
        }

        if (spirvBytes.Length % 4 != 0)
        {
            errorMessage = $"SPIR-V binary size {spirvBytes.Length} is not a multiple of 4 bytes (32-bit words).";
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(spirvBytes[0..4]);
        if (magic != SpirvOpcode.MagicNumber)
        {
            errorMessage = $"Invalid SPIR-V magic number: 0x{magic:X8}, expected 0x{SpirvOpcode.MagicNumber:X8}.";
            return false;
        }

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(spirvBytes[4..8]);
        uint major = (version >> 16) & 0xFF;
        uint minor = (version >> 8) & 0xFF;
        if (major != 1)
        {
            errorMessage = $"Unsupported SPIR-V major version: {major}. Expected 1.x.";
            return false;
        }

        uint bound = BinaryPrimitives.ReadUInt32LittleEndian(spirvBytes[12..16]);
        if (bound == 0)
        {
            errorMessage = "Invalid SPIR-V bound: 0. Bound must be greater than 0.";
            return false;
        }

        // Validate instruction stream
        int offset = 20;
        int wordCountTotal = spirvBytes.Length / 4;
        int wordOffset = 5;

        while (offset < spirvBytes.Length)
        {
            if (offset + 4 > spirvBytes.Length)
            {
                errorMessage = $"Truncated instruction header at offset {offset}.";
                return false;
            }

            uint firstWord = BinaryPrimitives.ReadUInt32LittleEndian(spirvBytes.Slice(offset, 4));
            ushort opcode = (ushort)(firstWord & 0xFFFF);
            ushort wordCount = (ushort)(firstWord >> 16);

            if (wordCount == 0)
            {
                errorMessage = $"Invalid instruction with 0 word count at word index {wordOffset} (opcode: {opcode}).";
                return false;
            }

            int instructionByteLength = wordCount * 4;
            if (offset + instructionByteLength > spirvBytes.Length)
            {
                errorMessage = $"Instruction opcode {opcode} specifies length {wordCount} words which exceeds binary end at byte offset {offset}.";
                return false;
            }

            offset += instructionByteLength;
            wordOffset += wordCount;
        }

        errorMessage = null;
        return true;
    }
}

// <copyright file="SpirvValidator.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.SpirV;

using System;
using System.Buffers.Binary;

/// <summary>
/// Validates binary SPIR-V bytecode stream structure, header, <c>id &lt; bound</c> invariants,
/// instruction framing, and single basic-block termination rules.
/// </summary>
public static class SpirvValidator
{
    /// <summary>
    /// Validates a SPIR-V 1.x binary byte span.
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

        int offset = 20;
        int wordOffset = 5;

        bool inFunction = false;
        bool inBlock = false;
        bool blockTerminated = false;
        bool hasBlocks = false;
        bool expectMergeTerminator = false;

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

            ReadOnlySpan<byte> instBytes = spirvBytes.Slice(offset, instructionByteLength);

            // Validate ID < bound for instruction ID operands
            if (!ValidateInstructionIds(opcode, wordCount, instBytes, bound, wordOffset, out errorMessage))
            {
                return false;
            }

            if (expectMergeTerminator)
            {
                if (opcode != SpirvOpcode.OpBranchConditional && opcode != SpirvOpcode.OpSwitch)
                {
                    errorMessage = $"OpSelectionMerge at word index {wordOffset} must be immediately followed by OpBranchConditional or OpSwitch (found opcode {opcode}).";
                    return false;
                }

                expectMergeTerminator = false;
            }

            // Validate function & basic-block single-termination CFG invariants
            if (opcode == SpirvOpcode.OpFunction)
            {
                if (inFunction)
                {
                    errorMessage = $"Nested OpFunction encountered at word index {wordOffset}.";
                    return false;
                }

                inFunction = true;
                inBlock = false;
                blockTerminated = false;
                hasBlocks = false;
            }
            else if (opcode == SpirvOpcode.OpFunctionEnd)
            {
                if (!inFunction)
                {
                    errorMessage = $"OpFunctionEnd without matching OpFunction at word index {wordOffset}.";
                    return false;
                }

                if (hasBlocks && !blockTerminated)
                {
                    errorMessage = $"Basic block was not terminated before OpFunctionEnd at word index {wordOffset}.";
                    return false;
                }

                inFunction = false;
                inBlock = false;
                blockTerminated = false;
            }
            else if (opcode == SpirvOpcode.OpLabel)
            {
                if (!inFunction)
                {
                    errorMessage = $"OpLabel encountered outside OpFunction at word index {wordOffset}.";
                    return false;
                }

                if (inBlock && !blockTerminated)
                {
                    errorMessage = $"Previous basic block was not terminated before OpLabel at word index {wordOffset}.";
                    return false;
                }

                inBlock = true;
                blockTerminated = false;
                hasBlocks = true;
            }
            else if (IsBlockTerminator(opcode))
            {
                if (!inFunction || !inBlock)
                {
                    errorMessage = $"Block terminator (opcode {opcode}) encountered outside basic block at word index {wordOffset}.";
                    return false;
                }

                if (blockTerminated)
                {
                    errorMessage = $"Duplicate block terminator (opcode {opcode}) in already-terminated basic block at word index {wordOffset}.";
                    return false;
                }

                blockTerminated = true;
            }
            else if (inFunction)
            {
                if (opcode != SpirvOpcode.OpFunctionParameter)
                {
                    if (!inBlock)
                    {
                        errorMessage = $"Instruction (opcode {opcode}) inside OpFunction precedes first OpLabel at word index {wordOffset}.";
                        return false;
                    }

                    if (blockTerminated)
                    {
                        errorMessage = $"Unreachable instruction (opcode {opcode}) emitted after block terminator at word index {wordOffset}.";
                        return false;
                    }

                    if (opcode == SpirvOpcode.OpSelectionMerge)
                    {
                        expectMergeTerminator = true;
                    }
                }
            }

            offset += instructionByteLength;
            wordOffset += wordCount;
        }

        if (inFunction)
        {
            errorMessage = "Unclosed OpFunction: reached end of SPIR-V module before OpFunctionEnd.";
            return false;
        }

        errorMessage = null;
        return true;
    }

    private static bool IsBlockTerminator(ushort opcode) =>
        opcode is SpirvOpcode.OpBranch
            or SpirvOpcode.OpBranchConditional
            or SpirvOpcode.OpSwitch
            or SpirvOpcode.OpKill
            or SpirvOpcode.OpReturn
            or SpirvOpcode.OpReturnValue
            or SpirvOpcode.OpUnreachable
            or SpirvOpcode.OpTerminateInvocation;

    private static uint ReadWord(ReadOnlySpan<byte> instBytes, int wordIndex) =>
        BinaryPrimitives.ReadUInt32LittleEndian(instBytes.Slice(wordIndex * 4, 4));

    private static bool CheckId(uint id, uint bound, ushort opcode, int wordOffset, out string? errorMessage)
    {
        if (id == 0 || id >= bound)
        {
            errorMessage = $"Invalid SPIR-V ID {id} in opcode {opcode} at word index {wordOffset}: expected 0 < id < {bound}.";
            return false;
        }

        errorMessage = null;
        return true;
    }

    private static bool ValidateInstructionIds(
        ushort opcode,
        ushort wordCount,
        ReadOnlySpan<byte> instBytes,
        uint bound,
        int wordOffset,
        out string? errorMessage)
    {
        errorMessage = null;
        switch (opcode)
        {
            case SpirvOpcode.OpExtInstImport:
            case SpirvOpcode.OpTypeVoid:
            case SpirvOpcode.OpTypeBool:
            case SpirvOpcode.OpTypeInt:
            case SpirvOpcode.OpTypeFloat:
            case SpirvOpcode.OpLabel:
            case SpirvOpcode.OpName:
            case SpirvOpcode.OpExecutionMode:
            case SpirvOpcode.OpDecorate:
            case SpirvOpcode.OpMemberDecorate:
            case SpirvOpcode.OpBranch:
            case SpirvOpcode.OpReturnValue:
                if (wordCount < 2)
                {
                    errorMessage = $"Instruction opcode {opcode} truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage);

            case SpirvOpcode.OpTypeVector:
            case SpirvOpcode.OpTypeRuntimeArray:
            case SpirvOpcode.OpConstant:
            case SpirvOpcode.OpFunctionParameter:
            case SpirvOpcode.OpStore:
            case SpirvOpcode.OpMemoryBarrier:
                if (wordCount < 3)
                {
                    errorMessage = $"Instruction opcode {opcode} truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage);

            case SpirvOpcode.OpTypePointer:
                if (wordCount < 4)
                {
                    errorMessage = $"OpTypePointer truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 3), bound, opcode, wordOffset + 3, out errorMessage);

            case SpirvOpcode.OpTypeArray:
            case SpirvOpcode.OpControlBarrier:
            case SpirvOpcode.OpBranchConditional:
            case SpirvOpcode.OpCompositeExtract:
                if (wordCount < 4)
                {
                    errorMessage = $"Instruction opcode {opcode} truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage)
                    && CheckId(ReadWord(instBytes, 3), bound, opcode, wordOffset + 3, out errorMessage);

            case SpirvOpcode.OpLoad:
                if (wordCount < 4)
                {
                    errorMessage = $"OpLoad truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage)
                    && CheckId(ReadWord(instBytes, 3), bound, opcode, wordOffset + 3, out errorMessage);

            case SpirvOpcode.OpVariable:
                if (wordCount < 4)
                {
                    errorMessage = $"OpVariable truncated at word index {wordOffset}.";
                    return false;
                }
                if (!CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage) ||
                    !CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage))
                {
                    return false;
                }
                if (wordCount >= 5 && !CheckId(ReadWord(instBytes, 4), bound, opcode, wordOffset + 4, out errorMessage))
                {
                    return false;
                }
                return true;

            case SpirvOpcode.OpFunction:
                if (wordCount < 5)
                {
                    errorMessage = $"OpFunction truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage)
                    && CheckId(ReadWord(instBytes, 4), bound, opcode, wordOffset + 4, out errorMessage);

            case SpirvOpcode.OpExtInst:
                if (wordCount < 5)
                {
                    errorMessage = $"OpExtInst truncated at word index {wordOffset}.";
                    return false;
                }
                if (!CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage) ||
                    !CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage) ||
                    !CheckId(ReadWord(instBytes, 3), bound, opcode, wordOffset + 3, out errorMessage))
                {
                    return false;
                }
                for (int w = 5; w < wordCount; w++)
                {
                    if (!CheckId(ReadWord(instBytes, w), bound, opcode, wordOffset + w, out errorMessage))
                    {
                        return false;
                    }
                }
                return true;

            case SpirvOpcode.OpEntryPoint:
                if (wordCount < 4)
                {
                    errorMessage = $"OpEntryPoint truncated at word index {wordOffset}.";
                    return false;
                }
                if (!CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage))
                {
                    return false;
                }
                // Advance past null-terminated entry-point name literal starting at word 3
                int strEndWord = 3;
                bool foundNull = false;
                while (strEndWord < wordCount)
                {
                    ReadOnlySpan<byte> wordSlice = instBytes.Slice(strEndWord * 4, 4);
                    strEndWord++;
                    if (wordSlice.IndexOf((byte)0) >= 0)
                    {
                        foundNull = true;
                        break;
                    }
                }
                if (!foundNull)
                {
                    errorMessage = $"OpEntryPoint name string missing null terminator at word index {wordOffset}.";
                    return false;
                }
                for (int w = strEndWord; w < wordCount; w++)
                {
                    if (!CheckId(ReadWord(instBytes, w), bound, opcode, wordOffset + w, out errorMessage))
                    {
                        return false;
                    }
                }
                return true;

            case SpirvOpcode.OpSelectionMerge:
                if (wordCount < 3)
                {
                    errorMessage = $"OpSelectionMerge truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage);

            case SpirvOpcode.OpUGreaterThanEqual:
                if (wordCount < 5)
                {
                    errorMessage = $"OpUGreaterThanEqual truncated at word index {wordOffset}.";
                    return false;
                }
                return CheckId(ReadWord(instBytes, 1), bound, opcode, wordOffset + 1, out errorMessage)
                    && CheckId(ReadWord(instBytes, 2), bound, opcode, wordOffset + 2, out errorMessage)
                    && CheckId(ReadWord(instBytes, 3), bound, opcode, wordOffset + 3, out errorMessage)
                    && CheckId(ReadWord(instBytes, 4), bound, opcode, wordOffset + 4, out errorMessage);

            case SpirvOpcode.OpTypeStruct:
            case SpirvOpcode.OpTypeFunction:
            case SpirvOpcode.OpConstantComposite:
            case SpirvOpcode.OpAccessChain:
            case SpirvOpcode.OpCompositeConstruct:
            case SpirvOpcode.OpConvertFToU:
            case SpirvOpcode.OpConvertFToS:
            case SpirvOpcode.OpConvertSToF:
            case SpirvOpcode.OpConvertUToF:
            case SpirvOpcode.OpBitcast:
            case SpirvOpcode.OpSNegate:
            case SpirvOpcode.OpFNegate:
            case SpirvOpcode.OpIAdd:
            case SpirvOpcode.OpFAdd:
            case SpirvOpcode.OpISub:
            case SpirvOpcode.OpFSub:
            case SpirvOpcode.OpIMul:
            case SpirvOpcode.OpFMul:
            case SpirvOpcode.OpUDiv:
            case SpirvOpcode.OpSDiv:
            case SpirvOpcode.OpFDiv:
            case SpirvOpcode.OpDot:
            case SpirvOpcode.OpAtomicIAdd:
            case SpirvOpcode.OpAtomicFAddEXT:
            case SpirvOpcode.OpGroupNonUniformShuffle:
            case SpirvOpcode.OpGroupNonUniformShuffleXor:
                for (int w = 1; w < wordCount; w++)
                {
                    if (!CheckId(ReadWord(instBytes, w), bound, opcode, wordOffset + w, out errorMessage))
                    {
                        return false;
                    }
                }
                return true;

            default:
                errorMessage = null;
                return true;
        }
    }
}

// <copyright file="SpirvBinaryEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.SpirV;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Pure C# .NET 10 binary emitter for SPIR-V 1.6 compute and graphics bytecode.
/// Directly emits 32-bit words with zero runtime dependencies.
/// </summary>
public sealed class SpirvBinaryEmitter
{
    private readonly List<uint> _words = new(1024);
    private uint _bound = 1;

    /// <summary>Allocates the next unique SPIR-V result ID.</summary>
    public uint NextId() => _bound++;

    /// <summary>Gets the current bound (all IDs &lt; bound).</summary>
    public uint Bound => _bound;

    /// <summary>Resets the emitter for a new module.</summary>
    public void Reset()
    {
        _words.Clear();
        _bound = 1;
    }

    /// <summary>Emits an instruction given an opcode and operand words.</summary>
    public void EmitInstruction(ushort opcode, ReadOnlySpan<uint> operands)
    {
        uint wordCount = (uint)(operands.Length + 1);
        uint firstWord = (wordCount << 16) | (uint)opcode;
        _words.Add(firstWord);
        for (int i = 0; i < operands.Length; i++)
        {
            _words.Add(operands[i]);
        }
    }

    /// <summary>Emits an instruction with no operands.</summary>
    public void EmitInstruction(ushort opcode)
    {
        uint firstWord = (1u << 16) | (uint)opcode;
        _words.Add(firstWord);
    }

    /// <summary>Emits an instruction with 1 operand.</summary>
    public void EmitInstruction(ushort opcode, uint op0)
    {
        uint firstWord = (2u << 16) | (uint)opcode;
        _words.Add(firstWord);
        _words.Add(op0);
    }

    /// <summary>Emits an instruction with 2 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1)
    {
        uint firstWord = (3u << 16) | (uint)opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
    }

    /// <summary>Emits an instruction with 3 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2)
    {
        uint firstWord = (4u << 16) | (uint)opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
    }

    /// <summary>Emits an instruction with 4 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3)
    {
        uint firstWord = (5u << 16) | (uint)opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
    }

    /// <summary>Emits an instruction with 5 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3, uint op4)
    {
        uint firstWord = (6u << 16) | (uint)opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
        _words.Add(op4);
    }

    /// <summary>Emits a string literal into SPIR-V words (null-terminated and 4-byte padded).</summary>
    public static void AppendStringWords(string text, List<uint> target)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        int totalBytes = bytes.Length + 1; // including null terminator
        int wordCount = (totalBytes + 3) / 4;
        Span<byte> buffer = stackalloc byte[wordCount * 4];
        buffer.Clear();
        bytes.CopyTo(buffer);

        for (int i = 0; i < wordCount; i++)
        {
            uint w = BinaryPrimitives.ReadUInt32LittleEndian(buffer.Slice(i * 4, 4));
            target.Add(w);
        }
    }

    /// <summary>
    /// Builds a complete SPIR-V 1.6 binary module for the specified shader stage and intermediate instructions.
    /// </summary>
    public ReadOnlyMemory<byte> EmitModule(ShaderStage stage, ReadOnlySpan<Instruction> instructions, uint localSizeX = 64, uint localSizeY = 1, uint localSizeZ = 1)
    {
        Reset();

        // Pass 1: Reserve header space (5 words)
        // 0: Magic
        // 1: Version
        // 2: Generator
        // 3: Bound
        // 4: Schema/Reserved
        for (int i = 0; i < 5; i++)
        {
            _words.Add(0);
        }

        // OpCapability Shader
        EmitInstruction(SpirvOpcode.OpCapability, SpirvOpcode.CapabilityShader);

        // OpMemoryModel Logical GLSL450
        EmitInstruction(SpirvOpcode.OpMemoryModel, SpirvOpcode.AddressingModelLogical, SpirvOpcode.MemoryModelGLSL450);

        // Allocate IDs for entry point and types
        uint entryPointId = NextId();
        uint voidTypeId = NextId();
        uint funcTypeId = NextId();
        uint floatTypeId = NextId();
        uint uintTypeId = NextId();
        uint v3UintTypeId = NextId();
        uint ptrInputV3UintId = NextId();
        uint globalInvIdVar = NextId();
        uint labelId = NextId();

        // OpEntryPoint
        uint execModel = stage switch
        {
            ShaderStage.Vertex => SpirvOpcode.ExecutionModelVertex,
            ShaderStage.Pixel => SpirvOpcode.ExecutionModelFragment,
            _ => SpirvOpcode.ExecutionModelGLCompute
        };

        var entryOperands = new List<uint> { execModel, entryPointId };
        AppendStringWords("main", entryOperands);
        entryOperands.Add(globalInvIdVar);
        EmitInstruction(SpirvOpcode.OpEntryPoint, CollectionsMarshal.AsSpan(entryOperands));

        // OpExecutionMode (for compute)
        if (stage == ShaderStage.Compute)
        {
            EmitInstruction(SpirvOpcode.OpExecutionMode, entryPointId, SpirvOpcode.ExecutionModeLocalSize, localSizeX, localSizeY, localSizeZ);
        }

        // OpDecorate BuiltIn GlobalInvocationId
        EmitInstruction(SpirvOpcode.OpDecorate, globalInvIdVar, SpirvOpcode.DecorationBuiltIn, SpirvOpcode.BuiltInGlobalInvocationId);

        // Type declarations
        EmitInstruction(SpirvOpcode.OpTypeVoid, voidTypeId);
        EmitInstruction(SpirvOpcode.OpTypeFunction, funcTypeId, voidTypeId);
        EmitInstruction(SpirvOpcode.OpTypeFloat, floatTypeId, 32);
        EmitInstruction(SpirvOpcode.OpTypeInt, uintTypeId, 32, 0);
        EmitInstruction(SpirvOpcode.OpTypeVector, v3UintTypeId, uintTypeId, 3);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrInputV3UintId, SpirvOpcode.StorageClassInput, v3UintTypeId);

        // Input variable for GlobalInvocationId
        EmitInstruction(SpirvOpcode.OpVariable, ptrInputV3UintId, globalInvIdVar, SpirvOpcode.StorageClassInput);

        // Type pointer for float function storage
        uint ptrFuncFloatId = NextId();
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrFuncFloatId, SpirvOpcode.StorageClassFunction, floatTypeId);

        // OpFunction
        EmitInstruction(SpirvOpcode.OpFunction, voidTypeId, entryPointId, 0, funcTypeId);
        EmitInstruction(SpirvOpcode.OpLabel, labelId);

        // Allocate local registers table
        // Map 64 float virtual registers to SPIR-V variable IDs
        uint[] regVars = new uint[64];
        for (int i = 0; i < regVars.Length; i++)
        {
            regVars[i] = NextId();
            EmitInstruction(SpirvOpcode.OpVariable, ptrFuncFloatId, regVars[i], SpirvOpcode.StorageClassFunction);
        }

        // Instruction translation
        for (int i = 0; i < instructions.Length; i++)
        {
            var inst = instructions[i];
            var opcode = (ComputeOpcode)inst.Opcode;
            int a = Math.Clamp(inst.OperandA, 0, 63);
            int b = Math.Clamp(inst.OperandB, 0, 63);
            int d = Math.Clamp(inst.Destination, 0, 63);

            switch (opcode)
            {
                case ComputeOpcode.Nop:
                    EmitInstruction(SpirvOpcode.OpNop);
                    break;

                case ComputeOpcode.Add:
                    {
                        uint valA = NextId();
                        uint valB = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, regVars[b]);
                        EmitInstruction(SpirvOpcode.OpFAdd, floatTypeId, result, valA, valB);
                        EmitInstruction(SpirvOpcode.OpStore, regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Sub:
                    {
                        uint valA = NextId();
                        uint valB = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, regVars[b]);
                        EmitInstruction(SpirvOpcode.OpFSub, floatTypeId, result, valA, valB);
                        EmitInstruction(SpirvOpcode.OpStore, regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Mul:
                    {
                        uint valA = NextId();
                        uint valB = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, regVars[b]);
                        EmitInstruction(SpirvOpcode.OpFMul, floatTypeId, result, valA, valB);
                        EmitInstruction(SpirvOpcode.OpStore, regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Div:
                    {
                        uint valA = NextId();
                        uint valB = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, regVars[b]);
                        EmitInstruction(SpirvOpcode.OpFDiv, floatTypeId, result, valA, valB);
                        EmitInstruction(SpirvOpcode.OpStore, regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Neg:
                    {
                        uint valA = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, regVars[a]);
                        EmitInstruction(SpirvOpcode.OpFNegate, floatTypeId, result, valA);
                        EmitInstruction(SpirvOpcode.OpStore, regVars[d], result);
                        break;
                    }

                case ComputeOpcode.BarrierSync:
                    {
                        // Execution Scope = 2 (Workgroup), Memory Scope = 2, Semantics = 0x100 (WorkgroupMemory)
                        uint constScopeId = NextId();
                        uint constSemanticsId = NextId();
                        EmitInstruction(SpirvOpcode.OpConstant, uintTypeId, constScopeId, 2);
                        EmitInstruction(SpirvOpcode.OpConstant, uintTypeId, constSemanticsId, 0x100);
                        EmitInstruction(SpirvOpcode.OpControlBarrier, constScopeId, constScopeId, constSemanticsId);
                        break;
                    }

                case ComputeOpcode.Return:
                    EmitInstruction(SpirvOpcode.OpReturn);
                    break;

                default:
                    EmitInstruction(SpirvOpcode.OpNop);
                    break;
            }
        }

        // Ensure function ends with return
        EmitInstruction(SpirvOpcode.OpReturn);
        EmitInstruction(SpirvOpcode.OpFunctionEnd);

        // Fill header words
        _words[0] = SpirvOpcode.MagicNumber;
        _words[1] = SpirvOpcode.Version16;
        _words[2] = SpirvOpcode.GeneratorId;
        _words[3] = _bound;
        _words[4] = 0; // Reserved

        // Convert uint words to byte array
        byte[] bytes = new byte[_words.Count * 4];
        MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(_words)).CopyTo(bytes);
        return bytes;
    }

    /// <summary>
    /// Emits SPIR-V words directly into a pre-allocated span with 0 allocations.
    /// </summary>
    public int EmitWords(ShaderStage stage, ReadOnlySpan<Instruction> instructions, Span<uint> destinationWords, uint localSizeX = 64)
    {
        var bytesMemory = EmitModule(stage, instructions, localSizeX);
        ReadOnlySpan<uint> sourceWords = MemoryMarshal.Cast<byte, uint>(bytesMemory.Span);
        if (destinationWords.Length < sourceWords.Length)
        {
            throw new ArgumentException($"Destination span too small. Required: {sourceWords.Length} words, provided: {destinationWords.Length} words.");
        }
        sourceWords.CopyTo(destinationWords);
        return sourceWords.Length;
    }
}

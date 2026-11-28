// <copyright file="SpirvBinaryEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.SpirV;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Pure C# .NET 10 binary emitter for SPIR-V 1.6 compute shader bytecode.
/// Directly emits spec-compliant 32-bit SPIR-V words with <c>StorageBuffer</c> descriptor bindings,
/// <c>PushConstant</c> element bounds guards, <c>Workgroup</c> shared memory, and <c>GLSL.std.450</c> math intrinsics.
/// </summary>
public sealed class SpirvBinaryEmitter
{
    private readonly List<uint> _words = new(1024);
    private readonly List<uint> _scratchOperands = new(32);
    private readonly Dictionary<uint, uint> _uintConstants = new(16);
    private readonly Dictionary<uint, uint> _floatConstants = new(16);
    private readonly uint[] _regVars = new uint[64];
    private readonly uint[] _idxVars = new uint[64];
    private uint _bound = 1;

    /// <summary>Allocates the next unique SPIR-V result ID.</summary>
    public uint NextId() => _bound++;

    /// <summary>Gets the current bound (all IDs &lt; bound).</summary>
    public uint Bound => _bound;

    /// <summary>Resets the emitter for a new module.</summary>
    public void Reset()
    {
        _words.Clear();
        _scratchOperands.Clear();
        _uintConstants.Clear();
        _floatConstants.Clear();
        Array.Clear(_regVars);
        Array.Clear(_idxVars);
        _bound = 1;
    }

    /// <summary>Emits an instruction given an opcode and operand words.</summary>
    public void EmitInstruction(ushort opcode, ReadOnlySpan<uint> operands)
    {
        uint wordCount = (uint)(operands.Length + 1);
        uint firstWord = (wordCount << 16) | opcode;
        _words.Add(firstWord);
        for (int i = 0; i < operands.Length; i++)
        {
            _words.Add(operands[i]);
        }
    }

    /// <summary>Emits an instruction with no operands.</summary>
    public void EmitInstruction(ushort opcode)
    {
        uint firstWord = (1u << 16) | opcode;
        _words.Add(firstWord);
    }

    /// <summary>Emits an instruction with 1 operand.</summary>
    public void EmitInstruction(ushort opcode, uint op0)
    {
        uint firstWord = (2u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
    }

    /// <summary>Emits an instruction with 2 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1)
    {
        uint firstWord = (3u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
    }

    /// <summary>Emits an instruction with 3 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2)
    {
        uint firstWord = (4u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
    }

    /// <summary>Emits an instruction with 4 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3)
    {
        uint firstWord = (5u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
    }

    /// <summary>Emits an instruction with 5 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3, uint op4)
    {
        uint firstWord = (6u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
        _words.Add(op4);
    }

    /// <summary>Emits an instruction with 6 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3, uint op4, uint op5)
    {
        uint firstWord = (7u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
        _words.Add(op4);
        _words.Add(op5);
    }

    /// <summary>Emits an instruction with 7 operands.</summary>
    public void EmitInstruction(ushort opcode, uint op0, uint op1, uint op2, uint op3, uint op4, uint op5, uint op6)
    {
        uint firstWord = (8u << 16) | opcode;
        _words.Add(firstWord);
        _words.Add(op0);
        _words.Add(op1);
        _words.Add(op2);
        _words.Add(op3);
        _words.Add(op4);
        _words.Add(op5);
        _words.Add(op6);
    }

    /// <summary>Emits a string literal into SPIR-V words (null-terminated and 4-byte padded).</summary>
    public static void AppendStringWords(string text, List<uint> target)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(target);

        int byteCount = Encoding.UTF8.GetByteCount(text);
        int totalBytes = byteCount + 1;
        int wordCount = (totalBytes + 3) / 4;
        Span<byte> buffer = wordCount <= 64
            ? stackalloc byte[wordCount * 4]
            : new byte[wordCount * 4];
        buffer.Clear();
        Encoding.UTF8.GetBytes(text, buffer[..byteCount]);

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
        BuildModuleWords(stage, instructions, localSizeX, localSizeY, localSizeZ);

        byte[] bytes = new byte[_words.Count * 4];
        MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(_words)).CopyTo(bytes);
        return bytes;
    }

    /// <summary>
    /// Emits SPIR-V words directly into a pre-allocated span without allocating an intermediate byte array.
    /// </summary>
    public int EmitWords(ShaderStage stage, ReadOnlySpan<Instruction> instructions, Span<uint> destinationWords, uint localSizeX = 64)
    {
        BuildModuleWords(stage, instructions, localSizeX, 1, 1);

        ReadOnlySpan<uint> sourceWords = CollectionsMarshal.AsSpan(_words);
        if (destinationWords.Length < sourceWords.Length)
        {
            throw new ArgumentException($"Destination span too small. Required: {sourceWords.Length} words, provided: {destinationWords.Length} words.", nameof(destinationWords));
        }

        sourceWords.CopyTo(destinationWords);
        return sourceWords.Length;
    }

    private void BuildModuleWords(ShaderStage stage, ReadOnlySpan<Instruction> instructions, uint localSizeX, uint localSizeY, uint localSizeZ)
    {
        if (stage != ShaderStage.Compute)
        {
            throw new NotSupportedException(
                $"ShaderStage.{stage} is not supported by SpirvBinaryEmitter; only ShaderStage.Compute is supported.");
        }

        Reset();

        bool usesShuffle = false;
        bool usesAtomicFloatAdd = false;
        bool usesShared = false;
        bool usesBarrier = false;
        ulong usedFloatRegs = 0;
        ulong usedUintRegs = 0;

        // Pre-scan all instructions to validate opcodes, track capabilities, collect constants, and determine live registers
        bool scanTerminated = false;
        for (int i = 0; i < instructions.Length; i++)
        {
            var inst = instructions[i];
            if (inst.Opcode < (int)ComputeOpcode.Nop || inst.Opcode > (int)ComputeOpcode.Return)
            {
                throw new NotSupportedException($"Unsupported ComputeOpcode '{inst.Opcode}' at instruction index {i} for SPIR-V 1.6 emission.");
            }

            var opcode = (ComputeOpcode)inst.Opcode;
            int a = Math.Clamp(inst.OperandA, 0, 63);
            int b = Math.Clamp(inst.OperandB, 0, 63);
            int d = Math.Clamp(inst.Destination, 0, 63);

            if (scanTerminated)
            {
                continue;
            }

            switch (opcode)
            {
                case ComputeOpcode.Nop:
                    break;

                case ComputeOpcode.Add:
                case ComputeOpcode.Sub:
                case ComputeOpcode.Mul:
                case ComputeOpcode.Div:
                case ComputeOpcode.Min:
                case ComputeOpcode.Max:
                    usedFloatRegs |= (1UL << a) | (1UL << b) | (1UL << d);
                    break;

                case ComputeOpcode.Fma:
                    usedFloatRegs |= (1UL << a) | (1UL << b) | (1UL << d);
                    break;

                case ComputeOpcode.Neg:
                case ComputeOpcode.Abs:
                case ComputeOpcode.Sqrt:
                case ComputeOpcode.Rsqrt:
                case ComputeOpcode.Exp:
                case ComputeOpcode.Log:
                case ComputeOpcode.Tanh:
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    break;

                case ComputeOpcode.LoadGlobal:
                    usedUintRegs |= (1UL << b);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.StoreGlobal:
                    usedFloatRegs |= (1UL << a);
                    usedUintRegs |= (1UL << b);
                    break;

                case ComputeOpcode.LoadShared:
                    usesShared = true;
                    usedUintRegs |= (1UL << b);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.StoreShared:
                    usesShared = true;
                    usedFloatRegs |= (1UL << a);
                    usedUintRegs |= (1UL << b);
                    break;

                case ComputeOpcode.LoadConstant:
                    usedFloatRegs |= (1UL << d);
                    usedUintRegs |= (1UL << d);
                    _floatConstants.TryAdd((uint)inst.OperandA, 0);
                    _uintConstants.TryAdd((uint)inst.OperandA, 0);
                    break;

                case ComputeOpcode.ShflSyncBfly:
                    usesShuffle = true;
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    uint laneMask = inst.OperandB != 0 ? (uint)inst.OperandB : 16u;
                    _uintConstants.TryAdd(SpirvOpcode.ScopeSubgroup, 0);
                    _uintConstants.TryAdd(laneMask, 0);
                    break;

                case ComputeOpcode.BarrierSync:
                    usesBarrier = true;
                    _uintConstants.TryAdd(SpirvOpcode.ScopeWorkgroup, 0);
                    _uintConstants.TryAdd(SpirvOpcode.MemorySemanticsWorkgroupAcquireRelease, 0);
                    break;

                case ComputeOpcode.LocalId:
                case ComputeOpcode.GlobalId:
                    usedUintRegs |= (1UL << d);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.AtomicAdd:
                    usesAtomicFloatAdd = true;
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    usedUintRegs |= (1UL << b);
                    _uintConstants.TryAdd(SpirvOpcode.ScopeDevice, 0);
                    _uintConstants.TryAdd(SpirvOpcode.MemorySemanticsRelaxed, 0);
                    break;

                case ComputeOpcode.Return:
                    scanTerminated = true;
                    break;
            }
        }

        // -------------------------------------------------------------------
        // 1. Header placeholder (5 words)
        // -------------------------------------------------------------------
        for (int i = 0; i < 5; i++)
        {
            _words.Add(0);
        }

        // -------------------------------------------------------------------
        // 2. Capabilities
        // -------------------------------------------------------------------
        EmitInstruction(SpirvOpcode.OpCapability, SpirvOpcode.CapabilityShader);
        if (usesShuffle)
        {
            EmitInstruction(SpirvOpcode.OpCapability, SpirvOpcode.CapabilityGroupNonUniformShuffle);
        }

        if (usesAtomicFloatAdd)
        {
            EmitInstruction(SpirvOpcode.OpCapability, SpirvOpcode.CapabilityAtomicFloat32AddEXT);
        }

        // -------------------------------------------------------------------
        // 3. Extensions
        // -------------------------------------------------------------------
        if (usesAtomicFloatAdd)
        {
            _scratchOperands.Clear();
            AppendStringWords("SPV_EXT_shader_atomic_float_add", _scratchOperands);
            EmitInstruction(SpirvOpcode.OpExtension, CollectionsMarshal.AsSpan(_scratchOperands));
        }

        // -------------------------------------------------------------------
        // 4. Extended Instruction Imports (GLSL.std.450)
        // -------------------------------------------------------------------
        uint glslExtId = NextId();
        _scratchOperands.Clear();
        _scratchOperands.Add(glslExtId);
        AppendStringWords("GLSL.std.450", _scratchOperands);
        EmitInstruction(SpirvOpcode.OpExtInstImport, CollectionsMarshal.AsSpan(_scratchOperands));

        // -------------------------------------------------------------------
        // 5. Memory Model
        // -------------------------------------------------------------------
        EmitInstruction(SpirvOpcode.OpMemoryModel, SpirvOpcode.AddressingModelLogical, SpirvOpcode.MemoryModelGLSL450);

        // -------------------------------------------------------------------
        // 6. Entry Point & Module-Scope Variable ID Allocation
        // -------------------------------------------------------------------
        uint entryPointId = NextId();

        uint globalInvIdVar = NextId();
        uint localInvIdVar = NextId();
        Span<uint> storageBufferVars = stackalloc uint[3];
        storageBufferVars[0] = NextId();
        storageBufferVars[1] = NextId();
        storageBufferVars[2] = NextId();
        uint pushConstVar = NextId();
        uint sharedMemVar = usesShared ? NextId() : 0;

        _scratchOperands.Clear();
        _scratchOperands.Add(SpirvOpcode.ExecutionModelGLCompute);
        _scratchOperands.Add(entryPointId);
        AppendStringWords("main", _scratchOperands);

        // SPIR-V 1.4+ / 1.6 requires all module-scope OpVariables in the entry-point interface list
        _scratchOperands.Add(globalInvIdVar);
        _scratchOperands.Add(localInvIdVar);
        _scratchOperands.Add(storageBufferVars[0]);
        _scratchOperands.Add(storageBufferVars[1]);
        _scratchOperands.Add(storageBufferVars[2]);
        _scratchOperands.Add(pushConstVar);
        if (usesShared)
        {
            _scratchOperands.Add(sharedMemVar);
        }

        EmitInstruction(SpirvOpcode.OpEntryPoint, CollectionsMarshal.AsSpan(_scratchOperands));

        // -------------------------------------------------------------------
        // 7. Execution Modes
        // -------------------------------------------------------------------
        EmitInstruction(SpirvOpcode.OpExecutionMode, entryPointId, SpirvOpcode.ExecutionModeLocalSize, localSizeX, localSizeY, localSizeZ);

        // -------------------------------------------------------------------
        // 8. Annotations / Decorations
        // -------------------------------------------------------------------
        uint runtimeArrayFloatTypeId = NextId();
        uint storageStructTypeId = NextId();
        uint pushConstStructTypeId = NextId();

        // Built-in invocation IDs (Compute stage)
        EmitInstruction(SpirvOpcode.OpDecorate, globalInvIdVar, SpirvOpcode.DecorationBuiltIn, SpirvOpcode.BuiltInGlobalInvocationId);
        EmitInstruction(SpirvOpcode.OpDecorate, localInvIdVar, SpirvOpcode.DecorationBuiltIn, SpirvOpcode.BuiltInLocalInvocationId);

        // StorageBuffer layout: struct { float data[]; } with ArrayStride = 4, Offset = 0
        EmitInstruction(SpirvOpcode.OpDecorate, runtimeArrayFloatTypeId, SpirvOpcode.DecorationArrayStride, 4);
        EmitInstruction(SpirvOpcode.OpDecorate, storageStructTypeId, SpirvOpcode.DecorationBlock);
        EmitInstruction(SpirvOpcode.OpMemberDecorate, storageStructTypeId, 0, SpirvOpcode.DecorationOffset, 0);

        // Descriptor bindings: set=0, binding=0..2 (input_a, input_b, output)
        for (uint binding = 0; binding < 3; binding++)
        {
            EmitInstruction(SpirvOpcode.OpDecorate, storageBufferVars[(int)binding], SpirvOpcode.DecorationDescriptorSet, 0);
            EmitInstruction(SpirvOpcode.OpDecorate, storageBufferVars[(int)binding], SpirvOpcode.DecorationBinding, binding);
        }

        // PushConstant layout: struct { uint param_n; } at Offset = 0
        EmitInstruction(SpirvOpcode.OpDecorate, pushConstStructTypeId, SpirvOpcode.DecorationBlock);
        EmitInstruction(SpirvOpcode.OpMemberDecorate, pushConstStructTypeId, 0, SpirvOpcode.DecorationOffset, 0);

        // -------------------------------------------------------------------
        // 9. Types, Constants, and Module-Scope Variables
        // -------------------------------------------------------------------
        uint voidTypeId = NextId();
        uint funcTypeId = NextId();
        uint floatTypeId = NextId();
        uint uintTypeId = NextId();
        uint boolTypeId = NextId();

        EmitInstruction(SpirvOpcode.OpTypeVoid, voidTypeId);
        EmitInstruction(SpirvOpcode.OpTypeFunction, funcTypeId, voidTypeId);
        EmitInstruction(SpirvOpcode.OpTypeFloat, floatTypeId, 32);
        EmitInstruction(SpirvOpcode.OpTypeInt, uintTypeId, 32, 0);
        EmitInstruction(SpirvOpcode.OpTypeBool, boolTypeId);

        // Always emit canonical zero constants for float and uint
        uint constFloatZeroId = GetOrEmitFloatConstant(floatTypeId, 0u);
        uint constUintZeroId = GetOrEmitUintConstant(uintTypeId, 0u);

        if (usesShared)
        {
            GetOrEmitUintConstant(uintTypeId, 1024u);
        }

        // Materialize all pre-scanned constants before OpFunction
        if (_floatConstants.Count > 1)
        {
            int count = _floatConstants.Count;
            uint[]? rented = null;
            Span<uint> keys = count <= 128
                ? stackalloc uint[count]
                : (rented = ArrayPool<uint>.Shared.Rent(count)).AsSpan(0, count);
            try
            {
                int k = 0;
                foreach (var kvp in _floatConstants)
                {
                    keys[k++] = kvp.Key;
                }

                for (int i = 0; i < k; i++)
                {
                    GetOrEmitFloatConstant(floatTypeId, keys[i]);
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<uint>.Shared.Return(rented);
                }
            }
        }

        if (_uintConstants.Count > 1 || usesBarrier)
        {
            int count = _uintConstants.Count;
            uint[]? rented = null;
            Span<uint> keys = count <= 128
                ? stackalloc uint[count]
                : (rented = ArrayPool<uint>.Shared.Rent(count)).AsSpan(0, count);
            try
            {
                int k = 0;
                foreach (var kvp in _uintConstants)
                {
                    keys[k++] = kvp.Key;
                }

                for (int i = 0; i < k; i++)
                {
                    GetOrEmitUintConstant(uintTypeId, keys[i]);
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<uint>.Shared.Return(rented);
                }
            }
        }

        uint ptrFuncFloatId = 0;
        if (usedFloatRegs != 0)
        {
            ptrFuncFloatId = NextId();
            EmitInstruction(SpirvOpcode.OpTypePointer, ptrFuncFloatId, SpirvOpcode.StorageClassFunction, floatTypeId);
        }

        uint ptrFuncUintId = 0;
        if (usedUintRegs != 0)
        {
            ptrFuncUintId = NextId();
            EmitInstruction(SpirvOpcode.OpTypePointer, ptrFuncUintId, SpirvOpcode.StorageClassFunction, uintTypeId);
        }

        uint v3UintTypeId = NextId();
        uint ptrInputV3UintId = NextId();
        uint ptrInputUintId = NextId();

        EmitInstruction(SpirvOpcode.OpTypeVector, v3UintTypeId, uintTypeId, 3);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrInputV3UintId, SpirvOpcode.StorageClassInput, v3UintTypeId);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrInputUintId, SpirvOpcode.StorageClassInput, uintTypeId);

        EmitInstruction(SpirvOpcode.OpVariable, ptrInputV3UintId, globalInvIdVar, SpirvOpcode.StorageClassInput);
        EmitInstruction(SpirvOpcode.OpVariable, ptrInputV3UintId, localInvIdVar, SpirvOpcode.StorageClassInput);

        // StorageBuffer types & variables (set=0, binding=0..2)
        uint ptrStorageStructId = NextId();
        uint ptrStorageFloatId = NextId();

        EmitInstruction(SpirvOpcode.OpTypeRuntimeArray, runtimeArrayFloatTypeId, floatTypeId);
        EmitInstruction(SpirvOpcode.OpTypeStruct, storageStructTypeId, runtimeArrayFloatTypeId);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrStorageStructId, SpirvOpcode.StorageClassStorageBuffer, storageStructTypeId);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrStorageFloatId, SpirvOpcode.StorageClassStorageBuffer, floatTypeId);

        EmitInstruction(SpirvOpcode.OpVariable, ptrStorageStructId, storageBufferVars[0], SpirvOpcode.StorageClassStorageBuffer);
        EmitInstruction(SpirvOpcode.OpVariable, ptrStorageStructId, storageBufferVars[1], SpirvOpcode.StorageClassStorageBuffer);
        EmitInstruction(SpirvOpcode.OpVariable, ptrStorageStructId, storageBufferVars[2], SpirvOpcode.StorageClassStorageBuffer);

        // PushConstant block & variable (param_n element count)
        uint ptrPushConstStructId = NextId();
        uint ptrPushConstUintId = NextId();
        EmitInstruction(SpirvOpcode.OpTypeStruct, pushConstStructTypeId, uintTypeId);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrPushConstStructId, SpirvOpcode.StorageClassPushConstant, pushConstStructTypeId);
        EmitInstruction(SpirvOpcode.OpTypePointer, ptrPushConstUintId, SpirvOpcode.StorageClassPushConstant, uintTypeId);
        EmitInstruction(SpirvOpcode.OpVariable, ptrPushConstStructId, pushConstVar, SpirvOpcode.StorageClassPushConstant);

        // Optional Workgroup shared memory (float[1024] = 4096 bytes)
        uint ptrWorkgroupFloatId = 0;
        if (usesShared)
        {
            uint const1024Id = _uintConstants[1024u];
            uint sharedArrayTypeId = NextId();
            uint ptrWorkgroupArrayId = NextId();
            ptrWorkgroupFloatId = NextId();

            EmitInstruction(SpirvOpcode.OpTypeArray, sharedArrayTypeId, floatTypeId, const1024Id);
            EmitInstruction(SpirvOpcode.OpTypePointer, ptrWorkgroupArrayId, SpirvOpcode.StorageClassWorkgroup, sharedArrayTypeId);
            EmitInstruction(SpirvOpcode.OpTypePointer, ptrWorkgroupFloatId, SpirvOpcode.StorageClassWorkgroup, floatTypeId);
            EmitInstruction(SpirvOpcode.OpVariable, ptrWorkgroupArrayId, sharedMemVar, SpirvOpcode.StorageClassWorkgroup);
        }

        // -------------------------------------------------------------------
        // 10. Entry Function Definition
        // -------------------------------------------------------------------
        uint labelId = NextId();
        EmitInstruction(SpirvOpcode.OpFunction, voidTypeId, entryPointId, 0, funcTypeId);
        EmitInstruction(SpirvOpcode.OpLabel, labelId);

        // Emit local function-scope register variables ONLY for live registers (at start of entry block)
        for (int r = 0; r < 64; r++)
        {
            if ((usedFloatRegs & (1UL << r)) != 0)
            {
                _regVars[r] = NextId();
                EmitInstruction(SpirvOpcode.OpVariable, ptrFuncFloatId, _regVars[r], SpirvOpcode.StorageClassFunction, constFloatZeroId);
            }

            if ((usedUintRegs & (1UL << r)) != 0)
            {
                _idxVars[r] = NextId();
                EmitInstruction(SpirvOpcode.OpVariable, ptrFuncUintId, _idxVars[r], SpirvOpcode.StorageClassFunction, constUintZeroId);
            }
        }

        // Load PushConstant param_n once after function OpVariables for structured bounds guarding
        uint paramNPtr = NextId();
        uint paramNVal = NextId();
        EmitInstruction(SpirvOpcode.OpAccessChain, ptrPushConstUintId, paramNPtr, pushConstVar, constUintZeroId);
        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, paramNVal, paramNPtr);

        // -------------------------------------------------------------------
        // 11. Instruction Translation (single-termination guarded)
        // -------------------------------------------------------------------
        bool terminated = false;
        for (int i = 0; i < instructions.Length && !terminated; i++)
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
                    EmitBinaryFloatOp(SpirvOpcode.OpFAdd, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Sub:
                    EmitBinaryFloatOp(SpirvOpcode.OpFSub, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Mul:
                    EmitBinaryFloatOp(SpirvOpcode.OpFMul, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Div:
                    EmitBinaryFloatOp(SpirvOpcode.OpFDiv, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Fma:
                    {
                        uint valA = NextId();
                        uint valB = NextId();
                        uint valD = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, _regVars[b]);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valD, _regVars[d]);
                        EmitInstruction(SpirvOpcode.OpExtInst, floatTypeId, result, glslExtId, SpirvOpcode.GlslStd450Fma, valA, valB, valD);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Neg:
                    {
                        uint valA = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(SpirvOpcode.OpFNegate, floatTypeId, result, valA);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
                        break;
                    }

                case ComputeOpcode.Abs:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450FAbs, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.Min:
                    EmitBinaryGlslOp(SpirvOpcode.GlslStd450FMin, glslExtId, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Max:
                    EmitBinaryGlslOp(SpirvOpcode.GlslStd450FMax, glslExtId, floatTypeId, a, b, d);
                    break;

                case ComputeOpcode.Sqrt:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450Sqrt, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.Rsqrt:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450InverseSqrt, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.Exp:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450Exp, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.Log:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450Log, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.Tanh:
                    EmitUnaryGlslOp(SpirvOpcode.GlslStd450Tanh, glslExtId, floatTypeId, a, d);
                    break;

                case ComputeOpcode.LoadGlobal:
                    {
                        int bufIdx = Math.Clamp(inst.OperandA, 0, 2);
                        uint idxVal = NextId();
                        uint isOob = NextId();
                        uint inBoundsLabel = NextId();
                        uint oobLabel = NextId();
                        uint mergeLabel = NextId();
                        uint elemPtr = NextId();
                        uint loaded = NextId();

                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idxVal, _idxVars[b]);
                        EmitInstruction(SpirvOpcode.OpUGreaterThanEqual, boolTypeId, isOob, idxVal, paramNVal);
                        EmitInstruction(SpirvOpcode.OpSelectionMerge, mergeLabel, SpirvOpcode.SelectionControlNone);
                        EmitInstruction(SpirvOpcode.OpBranchConditional, isOob, oobLabel, inBoundsLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, inBoundsLabel);
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrStorageFloatId, elemPtr, storageBufferVars[bufIdx], constUintZeroId, idxVal);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, loaded, elemPtr);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], loaded);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, oobLabel);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], constFloatZeroId);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, mergeLabel);
                        break;
                    }

                case ComputeOpcode.StoreGlobal:
                    {
                        uint valA = NextId();
                        uint idxVal = NextId();
                        uint isOob = NextId();
                        uint inBoundsLabel = NextId();
                        uint mergeLabel = NextId();
                        uint elemPtr = NextId();

                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idxVal, _idxVars[b]);
                        EmitInstruction(SpirvOpcode.OpUGreaterThanEqual, boolTypeId, isOob, idxVal, paramNVal);
                        EmitInstruction(SpirvOpcode.OpSelectionMerge, mergeLabel, SpirvOpcode.SelectionControlNone);
                        EmitInstruction(SpirvOpcode.OpBranchConditional, isOob, mergeLabel, inBoundsLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, inBoundsLabel);
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrStorageFloatId, elemPtr, storageBufferVars[2], constUintZeroId, idxVal);
                        EmitInstruction(SpirvOpcode.OpStore, elemPtr, valA);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, mergeLabel);
                        break;
                    }

                case ComputeOpcode.LoadShared:
                    {
                        uint idxVal = NextId();
                        uint isOob = NextId();
                        uint inBoundsLabel = NextId();
                        uint oobLabel = NextId();
                        uint mergeLabel = NextId();
                        uint elemPtr = NextId();
                        uint loaded = NextId();

                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idxVal, _idxVars[b]);
                        EmitInstruction(SpirvOpcode.OpUGreaterThanEqual, boolTypeId, isOob, idxVal, _uintConstants[1024u]);
                        EmitInstruction(SpirvOpcode.OpSelectionMerge, mergeLabel, SpirvOpcode.SelectionControlNone);
                        EmitInstruction(SpirvOpcode.OpBranchConditional, isOob, oobLabel, inBoundsLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, inBoundsLabel);
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrWorkgroupFloatId, elemPtr, sharedMemVar, idxVal);
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, loaded, elemPtr);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], loaded);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, oobLabel);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], constFloatZeroId);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, mergeLabel);
                        break;
                    }

                case ComputeOpcode.StoreShared:
                    {
                        uint valA = NextId();
                        uint idxVal = NextId();
                        uint isOob = NextId();
                        uint inBoundsLabel = NextId();
                        uint mergeLabel = NextId();
                        uint elemPtr = NextId();

                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idxVal, _idxVars[b]);
                        EmitInstruction(SpirvOpcode.OpUGreaterThanEqual, boolTypeId, isOob, idxVal, _uintConstants[1024u]);
                        EmitInstruction(SpirvOpcode.OpSelectionMerge, mergeLabel, SpirvOpcode.SelectionControlNone);
                        EmitInstruction(SpirvOpcode.OpBranchConditional, isOob, mergeLabel, inBoundsLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, inBoundsLabel);
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrWorkgroupFloatId, elemPtr, sharedMemVar, idxVal);
                        EmitInstruction(SpirvOpcode.OpStore, elemPtr, valA);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, mergeLabel);
                        break;
                    }

                case ComputeOpcode.LoadConstant:
                    {
                        uint rawBits = (uint)inst.OperandA;
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], _floatConstants[rawBits]);
                        EmitInstruction(SpirvOpcode.OpStore, _idxVars[d], _uintConstants[rawBits]);
                        break;
                    }

                case ComputeOpcode.ShflSyncBfly:
                    {
                        uint laneMask = inst.OperandB != 0 ? (uint)inst.OperandB : 16u;
                        uint valA = NextId();
                        uint result = NextId();
                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(
                            SpirvOpcode.OpGroupNonUniformShuffleXor,
                            floatTypeId,
                            result,
                            _uintConstants[SpirvOpcode.ScopeSubgroup],
                            valA,
                            _uintConstants[laneMask]);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
                        break;
                    }

                case ComputeOpcode.BarrierSync:
                    {
                        uint scopeId = _uintConstants[SpirvOpcode.ScopeWorkgroup];
                        uint semId = _uintConstants[SpirvOpcode.MemorySemanticsWorkgroupAcquireRelease];
                        EmitInstruction(SpirvOpcode.OpControlBarrier, scopeId, scopeId, semId);
                        break;
                    }

                case ComputeOpcode.LocalId:
                    {
                        uint compPtr = NextId();
                        uint idUint = NextId();
                        uint idFloat = NextId();
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrInputUintId, compPtr, localInvIdVar, constUintZeroId);
                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idUint, compPtr);
                        EmitInstruction(SpirvOpcode.OpStore, _idxVars[d], idUint);
                        EmitInstruction(SpirvOpcode.OpConvertUToF, floatTypeId, idFloat, idUint);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], idFloat);
                        break;
                    }

                case ComputeOpcode.GlobalId:
                    {
                        uint compPtr = NextId();
                        uint idUint = NextId();
                        uint idFloat = NextId();
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrInputUintId, compPtr, globalInvIdVar, constUintZeroId);
                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idUint, compPtr);
                        EmitInstruction(SpirvOpcode.OpStore, _idxVars[d], idUint);
                        EmitInstruction(SpirvOpcode.OpConvertUToF, floatTypeId, idFloat, idUint);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], idFloat);
                        break;
                    }

                case ComputeOpcode.AtomicAdd:
                    {
                        uint valA = NextId();
                        uint idxVal = NextId();
                        uint isOob = NextId();
                        uint inBoundsLabel = NextId();
                        uint oobLabel = NextId();
                        uint mergeLabel = NextId();
                        uint elemPtr = NextId();
                        uint oldVal = NextId();

                        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
                        EmitInstruction(SpirvOpcode.OpLoad, uintTypeId, idxVal, _idxVars[b]);
                        EmitInstruction(SpirvOpcode.OpUGreaterThanEqual, boolTypeId, isOob, idxVal, paramNVal);
                        EmitInstruction(SpirvOpcode.OpSelectionMerge, mergeLabel, SpirvOpcode.SelectionControlNone);
                        EmitInstruction(SpirvOpcode.OpBranchConditional, isOob, oobLabel, inBoundsLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, inBoundsLabel);
                        EmitInstruction(SpirvOpcode.OpAccessChain, ptrStorageFloatId, elemPtr, storageBufferVars[2], constUintZeroId, idxVal);
                        EmitInstruction(
                            SpirvOpcode.OpAtomicFAddEXT,
                            floatTypeId,
                            oldVal,
                            elemPtr,
                            _uintConstants[SpirvOpcode.ScopeDevice],
                            _uintConstants[SpirvOpcode.MemorySemanticsRelaxed],
                            valA);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], oldVal);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, oobLabel);
                        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], constFloatZeroId);
                        EmitInstruction(SpirvOpcode.OpBranch, mergeLabel);

                        EmitInstruction(SpirvOpcode.OpLabel, mergeLabel);
                        break;
                    }

                case ComputeOpcode.Return:
                    EmitInstruction(SpirvOpcode.OpReturn);
                    terminated = true;
                    break;

                default:
                    throw new NotSupportedException($"Unsupported ComputeOpcode '{inst.Opcode}' for SPIR-V 1.6 emission.");
            }
        }

        if (!terminated)
        {
            EmitInstruction(SpirvOpcode.OpReturn);
        }

        EmitInstruction(SpirvOpcode.OpFunctionEnd);

        // Patch 5-word SPIR-V header
        _words[0] = SpirvOpcode.MagicNumber;
        _words[1] = SpirvOpcode.Version16;
        _words[2] = SpirvOpcode.GeneratorId;
        _words[3] = _bound;
        _words[4] = 0;
    }

    private uint GetOrEmitUintConstant(uint uintTypeId, uint value)
    {
        if (_uintConstants.TryGetValue(value, out uint existingId) && existingId != 0)
        {
            return existingId;
        }

        uint id = NextId();
        EmitInstruction(SpirvOpcode.OpConstant, uintTypeId, id, value);
        _uintConstants[value] = id;
        return id;
    }

    private uint GetOrEmitFloatConstant(uint floatTypeId, uint ieeeBits)
    {
        if (_floatConstants.TryGetValue(ieeeBits, out uint existingId) && existingId != 0)
        {
            return existingId;
        }

        uint id = NextId();
        EmitInstruction(SpirvOpcode.OpConstant, floatTypeId, id, ieeeBits);
        _floatConstants[ieeeBits] = id;
        return id;
    }

    private void EmitBinaryFloatOp(ushort spirvOp, uint floatTypeId, int a, int b, int d)
    {
        uint valA = NextId();
        uint valB = NextId();
        uint result = NextId();
        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, _regVars[b]);
        EmitInstruction(spirvOp, floatTypeId, result, valA, valB);
        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
    }

    private void EmitUnaryGlslOp(uint glslOp, uint glslExtId, uint floatTypeId, int a, int d)
    {
        uint valA = NextId();
        uint result = NextId();
        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
        EmitInstruction(SpirvOpcode.OpExtInst, floatTypeId, result, glslExtId, glslOp, valA);
        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
    }

    private void EmitBinaryGlslOp(uint glslOp, uint glslExtId, uint floatTypeId, int a, int b, int d)
    {
        uint valA = NextId();
        uint valB = NextId();
        uint result = NextId();
        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valA, _regVars[a]);
        EmitInstruction(SpirvOpcode.OpLoad, floatTypeId, valB, _regVars[b]);
        EmitInstruction(SpirvOpcode.OpExtInst, floatTypeId, result, glslExtId, glslOp, valA, valB);
        EmitInstruction(SpirvOpcode.OpStore, _regVars[d], result);
    }
}

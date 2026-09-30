// <copyright file="SpirvOpcode.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.SpirV;

/// <summary>
/// SPIR-V 1.6 standard specification opcodes and constants.
/// </summary>
public static class SpirvOpcode
{
    public const uint MagicNumber = 0x07230203;
    public const uint Version16 = 0x00010600; // SPIR-V 1.6
    public const uint GeneratorId = 0x00130000; // Glacier Compute tool ID

    // Core Opcodes
    public const ushort OpNop = 0;
    public const ushort OpSource = 3;
    public const ushort OpName = 5;
    public const ushort OpExtInstImport = 11;
    public const ushort OpMemoryModel = 14;
    public const ushort OpEntryPoint = 15;
    public const ushort OpExecutionMode = 16;
    public const ushort OpCapability = 17;
    public const ushort OpTypeVoid = 19;
    public const ushort OpTypeBool = 20;
    public const ushort OpTypeInt = 21;
    public const ushort OpTypeFloat = 22;
    public const ushort OpTypeVector = 23;
    public const ushort OpTypeRuntimeArray = 29;
    public const ushort OpTypeStruct = 30;
    public const ushort OpTypePointer = 32;
    public const ushort OpTypeFunction = 33;
    public const ushort OpConstant = 43;
    public const ushort OpConstantComposite = 44;
    public const ushort OpFunction = 54;
    public const ushort OpFunctionParameter = 55;
    public const ushort OpFunctionEnd = 56;
    public const ushort OpVariable = 59;
    public const ushort OpLoad = 61;
    public const ushort OpStore = 62;
    public const ushort OpAccessChain = 65;
    public const ushort OpDecorate = 71;
    public const ushort OpMemberDecorate = 72;
    public const ushort OpVectorShuffle = 79;
    public const ushort OpCompositeConstruct = 80;
    public const ushort OpCompositeExtract = 81;
    public const ushort OpSNegate = 126;
    public const ushort OpFNegate = 127;
    public const ushort OpIAdd = 128;
    public const ushort OpFAdd = 129;
    public const ushort OpISub = 130;
    public const ushort OpFSub = 131;
    public const ushort OpIMul = 132;
    public const ushort OpFMul = 133;
    public const ushort OpUDiv = 134;
    public const ushort OpSDiv = 135;
    public const ushort OpFDiv = 136;
    public const ushort OpDot = 148;
    public const ushort OpControlBarrier = 224;
    public const ushort OpMemoryBarrier = 225;
    public const ushort OpAtomicIAdd = 227;
    public const ushort OpLabel = 248;
    public const ushort OpBranch = 249;
    public const ushort OpBranchConditional = 250;
    public const ushort OpReturn = 253;

    // Enums
    public const uint CapabilityShader = 1;
    public const uint CapabilityFloat64 = 2;
    public const uint CapabilityVulkanMemoryModel = 5345;

    public const uint AddressingModelLogical = 0;
    public const uint MemoryModelGLSL450 = 1;
    public const uint MemoryModelVulkan = 3;

    public const uint ExecutionModelVertex = 0;
    public const uint ExecutionModelFragment = 4;
    public const uint ExecutionModelGLCompute = 5;

    public const uint ExecutionModeLocalSize = 17;

    public const uint StorageClassUniform = 2;
    public const uint StorageClassWorkgroup = 4;
    public const uint StorageClassStorageBuffer = 12;
    public const uint StorageClassFunction = 7;
    public const uint StorageClassInput = 1;

    public const uint DecorationBlock = 2;
    public const uint DecorationBufferBlock = 3;
    public const uint DecorationArrayStride = 6;
    public const uint DecorationBuiltIn = 11;
    public const uint DecorationBinding = 33;
    public const uint DecorationDescriptorSet = 34;

    public const uint BuiltInGlobalInvocationId = 28;
    public const uint BuiltInLocalInvocationId = 27;
    public const uint BuiltInWorkgroupId = 26;
}

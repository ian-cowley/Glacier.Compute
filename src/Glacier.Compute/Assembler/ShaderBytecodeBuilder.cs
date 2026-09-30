// <copyright file="ShaderBytecodeBuilder.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler;

using System;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Assembler.SpirV;

/// <summary>
/// Unified in-memory shader bytecode assembler implementing <see cref="IShaderBytecodeBuilder"/>.
/// Generates NVIDIA PTX 8.0 text, binary SPIR-V 1.6, and Direct3D 12 DXBC/DXIL without external toolchains.
/// </summary>
public sealed class ShaderBytecodeBuilder : IShaderBytecodeBuilder
{
    private readonly SpirvBinaryEmitter _spirvEmitter = new();

    /// <inheritdoc/>
    public ReadOnlyMemory<byte> BuildSpirV(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        return _spirvEmitter.EmitModule(stage, instructions);
    }

    /// <inheritdoc/>
    public ReadOnlyMemory<byte> BuildDxil(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        return DxbcContainerBuilder.BuildComputeShader(stage, instructions);
    }

    /// <inheritdoc/>
    public string BuildPtx(ReadOnlySpan<Instruction> instructions)
    {
        return PtxEmitter.Emit(instructions);
    }
}

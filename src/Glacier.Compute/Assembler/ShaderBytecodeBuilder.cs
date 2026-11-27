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
/// Generates NVIDIA PTX 8.0 text and binary SPIR-V 1.6 modules directly in pure C# without external toolchains.
/// Direct DXIL LLVM 3.7 synthesis is gated with <see cref="NotSupportedException"/>; use
/// <see cref="DxbcContainerBuilder.PackagePrecompiledShader"/> for precompiled DXIL blobs.
/// </summary>
public sealed class ShaderBytecodeBuilder : IShaderBytecodeBuilder
{
    /// <inheritdoc/>
    public ReadOnlyMemory<byte> BuildSpirV(ShaderStage stage, ReadOnlySpan<Instruction> instructions)
    {
        return new SpirvBinaryEmitter().EmitModule(stage, instructions);
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">
    /// Always thrown because direct LLVM 3.7 DXIL bitcode emission requires external DXC compilation and <c>dxil.dll</c> signing.
    /// </exception>
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

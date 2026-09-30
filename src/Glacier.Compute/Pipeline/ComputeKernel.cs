// <copyright file="ComputeKernel.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Pipeline;

using System;
using Glacier.Compute.Assembler;

/// <summary>
/// Compiled compute kernel container containing cross-backend bytecode representations.
/// </summary>
public sealed class ComputeKernel
{
    /// <summary>Gets the unique name of this kernel.</summary>
    public string Name { get; }

    /// <summary>Gets the target shader stage.</summary>
    public ShaderStage Stage { get; }

    /// <summary>Gets the NVIDIA PTX 8.0 assembly text, if built.</summary>
    public string? PtxAssembly { get; }

    /// <summary>Gets the SPIR-V 1.6 binary bytecode, if built.</summary>
    public ReadOnlyMemory<byte>? SpirvBytecode { get; }

    /// <summary>Gets the DirectX Container (DXBC) containing DXIL bitcode, if built.</summary>
    public ReadOnlyMemory<byte>? DxilContainer { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ComputeKernel"/> class.
    /// </summary>
    public ComputeKernel(
        string name,
        ShaderStage stage,
        string? ptxAssembly,
        ReadOnlyMemory<byte>? spirvBytecode,
        ReadOnlyMemory<byte>? dxilContainer)
    {
        Name = name;
        Stage = stage;
        PtxAssembly = ptxAssembly;
        SpirvBytecode = spirvBytecode;
        DxilContainer = dxilContainer;
    }
}

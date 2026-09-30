// <copyright file="ShaderPipeline.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Pipeline;

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Glacier.Compute.Assembler;

/// <summary>
/// Unified compute shader pipeline orchestrator. Manages instruction compilation,
/// bytecode translation across PTX, SPIR-V, and DXIL/DXBC, and kernel caching.
/// </summary>
public sealed class ShaderPipeline
{
    private readonly IShaderBytecodeBuilder _builder;
    private readonly ConcurrentDictionary<ulong, ComputeKernel> _kernelCache = new();

    /// <summary>Gets the underlying shader bytecode assembler.</summary>
    public IShaderBytecodeBuilder Assembler => _builder;

    /// <summary>Gets the number of cached compiled compute kernels.</summary>
    public int CachedKernelCount => _kernelCache.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="ShaderPipeline"/> class.
    /// </summary>
    /// <param name="builder">The bytecode builder (defaults to <see cref="ShaderBytecodeBuilder"/>).</param>
    public ShaderPipeline(IShaderBytecodeBuilder? builder = null)
    {
        _builder = builder ?? new ShaderBytecodeBuilder();
    }

    /// <summary>
    /// Compiles an instruction sequence into a <see cref="ComputeKernel"/> for all requested backends,
    /// caching the result by instruction hash for instant reuse.
    /// </summary>
    public ComputeKernel Compile(
        string kernelName,
        ShaderStage stage,
        ReadOnlySpan<Instruction> instructions,
        bool includePtx = true,
        bool includeSpirv = true,
        bool includeDxil = true)
    {
        ulong hash = ComputeInstructionsHash(instructions, stage);
        if (_kernelCache.TryGetValue(hash, out var existing))
        {
            return existing;
        }

        string? ptx = includePtx ? _builder.BuildPtx(instructions) : null;
        ReadOnlyMemory<byte>? spirv = includeSpirv ? _builder.BuildSpirV(stage, instructions) : null;
        ReadOnlyMemory<byte>? dxil = includeDxil ? _builder.BuildDxil(stage, instructions) : null;

        var kernel = new ComputeKernel(kernelName, stage, ptx, spirv, dxil);
        _kernelCache.TryAdd(hash, kernel);
        return kernel;
    }

    /// <summary>
    /// Computes a 64-bit non-cryptographic hash for the instruction sequence.
    /// </summary>
    public static ulong ComputeInstructionsHash(ReadOnlySpan<Instruction> instructions, ShaderStage stage)
    {
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(instructions);
        ulong hash = 14695981039346656037UL ^ (ulong)stage;
        for (int i = 0; i < bytes.Length; i++)
        {
            hash ^= bytes[i];
            hash *= 1099511628211UL;
        }
        return hash;
    }
}

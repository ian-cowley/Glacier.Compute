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
/// bytecode translation across PTX 8.0 and SPIR-V 1.6 (plus optional custom DXIL builders), and kernel caching.
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
    /// Compiles an instruction sequence into a <see cref="ComputeKernel"/> for the requested backends,
    /// caching the result by instruction and target-flag hash for instant reuse.
    /// Defaults <paramref name="includeDxil"/> to <see langword="false"/> because direct DXIL synthesis requires external DXC signing.
    /// </summary>
    public ComputeKernel Compile(
        string kernelName,
        ShaderStage stage,
        ReadOnlySpan<Instruction> instructions,
        bool includePtx = true,
        bool includeSpirv = true,
        bool includeDxil = false)
    {
        ArgumentNullException.ThrowIfNull(kernelName);

        ulong baseHash = ComputeInstructionsHash(instructions, stage);
        ulong flagsMask = (includePtx ? 1UL : 0UL) | (includeSpirv ? 2UL : 0UL) | (includeDxil ? 4UL : 0UL);
        ulong cacheKey = (baseHash ^ (flagsMask * 0x9E3779B97F4A7C15UL)) * 1099511628211UL;

        if (_kernelCache.TryGetValue(cacheKey, out var existing))
        {
            return existing;
        }

        string? ptx = includePtx ? _builder.BuildPtx(instructions) : null;
        ReadOnlyMemory<byte>? spirv = includeSpirv ? (ReadOnlyMemory<byte>?)_builder.BuildSpirV(stage, instructions) : null;
        ReadOnlyMemory<byte>? dxil = includeDxil ? (ReadOnlyMemory<byte>?)_builder.BuildDxil(stage, instructions) : null;

        var kernel = new ComputeKernel(kernelName, stage, ptx, spirv, dxil);
        return _kernelCache.GetOrAdd(cacheKey, kernel);
    }

    /// <summary>
    /// Computes a 64-bit FNV-1a non-cryptographic hash for the instruction sequence and stage.
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

// <copyright file="ShaderPipeline.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Pipeline;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Compute.Assembler;

/// <summary>
/// Unified compute shader pipeline orchestrator. Manages instruction compilation,
/// bytecode translation across PTX 8.0 and SPIR-V 1.6 (plus optional custom DXIL builders), and bounded LRU kernel caching.
/// </summary>
public sealed class ShaderPipeline
{
    /// <summary>Default maximum number of compiled kernels retained in the LRU cache.</summary>
    public const int DefaultMaxCachedKernels = 1024;

    private readonly IShaderBytecodeBuilder _builder;
    private readonly int _maxCachedKernels;
    private readonly Lock _cacheLock = new();
    private readonly Dictionary<KernelCacheKey, LinkedListNode<CacheEntry>> _kernelCache;
    private readonly LinkedList<CacheEntry> _lruList = new();

    /// <summary>Gets the underlying shader bytecode assembler.</summary>
    public IShaderBytecodeBuilder Assembler => _builder;

    /// <summary>Gets the maximum number of compiled kernels retained in the cache.</summary>
    public int MaxCachedKernels => _maxCachedKernels;

    /// <summary>Gets the number of cached compiled compute kernels.</summary>
    public int CachedKernelCount
    {
        get
        {
            lock (_cacheLock)
            {
                return _kernelCache.Count;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ShaderPipeline"/> class with <see cref="DefaultMaxCachedKernels"/> capacity.
    /// </summary>
    /// <param name="builder">The bytecode builder (defaults to <see cref="ShaderBytecodeBuilder"/>).</param>
    public ShaderPipeline(IShaderBytecodeBuilder? builder = null)
        : this(builder, DefaultMaxCachedKernels)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ShaderPipeline"/> class with an explicit LRU cache capacity.
    /// </summary>
    /// <param name="builder">The bytecode builder (defaults to <see cref="ShaderBytecodeBuilder"/>).</param>
    /// <param name="maxCachedKernels">Maximum number of compiled kernels to cache (must be positive).</param>
    public ShaderPipeline(IShaderBytecodeBuilder? builder, int maxCachedKernels)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxCachedKernels);
        _builder = builder ?? new ShaderBytecodeBuilder();
        _maxCachedKernels = maxCachedKernels;
        _kernelCache = new Dictionary<KernelCacheKey, LinkedListNode<CacheEntry>>(
            Math.Min(maxCachedKernels, 256),
            KernelKeyComparer.Instance);
    }

    /// <summary>
    /// Compiles an instruction sequence into a <see cref="ComputeKernel"/> for the requested backends,
    /// caching the result in a bounded LRU cache keyed by stage, target flags, and exact instruction sequence.
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

        byte flagsMask = (byte)((includePtx ? 1 : 0) | (includeSpirv ? 2 : 0) | (includeDxil ? 4 : 0));
        int hashCode = ComputeLookupHashCode(instructions, stage, flagsMask);
        var lookup = new KernelLookupKey(stage, flagsMask, instructions, hashCode);

        lock (_cacheLock)
        {
            var altLookup = _kernelCache.GetAlternateLookup<KernelLookupKey>();
            if (altLookup.TryGetValue(lookup, out var existingNode))
            {
                PromoteToFront(existingNode);
                return existingNode.Value.Kernel;
            }
        }

        // Compile outside the cache lock so compilation latency or NotSupportedException does not block other threads.
        string? ptx = includePtx ? _builder.BuildPtx(instructions) : null;
        ReadOnlyMemory<byte>? spirv = includeSpirv ? (ReadOnlyMemory<byte>?)_builder.BuildSpirV(stage, instructions) : null;
        ReadOnlyMemory<byte>? dxil = includeDxil ? (ReadOnlyMemory<byte>?)_builder.BuildDxil(stage, instructions) : null;

        var compiled = new ComputeKernel(kernelName, stage, ptx, spirv, dxil);

        lock (_cacheLock)
        {
            var altLookup = _kernelCache.GetAlternateLookup<KernelLookupKey>();
            if (altLookup.TryGetValue(lookup, out var racingNode))
            {
                PromoteToFront(racingNode);
                return racingNode.Value.Kernel;
            }

            KernelCacheKey ownedKey = KernelKeyComparer.Instance.Create(lookup);
            var entry = new CacheEntry(ownedKey, compiled);
            var newNode = new LinkedListNode<CacheEntry>(entry);
            _lruList.AddFirst(newNode);
            _kernelCache.Add(ownedKey, newNode);

            while (_kernelCache.Count > _maxCachedKernels && _lruList.Last is { } lruNode)
            {
                _lruList.RemoveLast();
                _kernelCache.Remove(lruNode.Value.Key);
            }

            return compiled;
        }
    }

    private void PromoteToFront(LinkedListNode<CacheEntry> node)
    {
        if (!ReferenceEquals(_lruList.First, node))
        {
            _lruList.Remove(node);
            _lruList.AddFirst(node);
        }
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

    private static int ComputeLookupHashCode(ReadOnlySpan<Instruction> instructions, ShaderStage stage, byte flagsMask)
    {
        ulong fnv = ComputeInstructionsHash(instructions, stage);
        ulong mixed = (fnv ^ (flagsMask * 0x9E3779B97F4A7C15UL)) * 1099511628211UL;
        return unchecked((int)(mixed ^ (mixed >> 32)));
    }

    private sealed record class CacheEntry(KernelCacheKey Key, ComputeKernel Kernel);

    private readonly struct KernelCacheKey : IEquatable<KernelCacheKey>
    {
        public readonly ShaderStage Stage;
        public readonly byte FlagsMask;
        public readonly Instruction[] Instructions;
        public readonly int HashCode;

        public KernelCacheKey(ShaderStage stage, byte flagsMask, Instruction[] instructions, int hashCode)
        {
            Stage = stage;
            FlagsMask = flagsMask;
            Instructions = instructions;
            HashCode = hashCode;
        }

        public bool Equals(KernelCacheKey other)
        {
            return Stage == other.Stage
                && FlagsMask == other.FlagsMask
                && HashCode == other.HashCode
                && Instructions.AsSpan().SequenceEqual(other.Instructions);
        }

        public override bool Equals(object? obj) => obj is KernelCacheKey other && Equals(other);

        public override int GetHashCode() => HashCode;
    }

    private readonly ref struct KernelLookupKey
    {
        public readonly ShaderStage Stage;
        public readonly byte FlagsMask;
        public readonly ReadOnlySpan<Instruction> Instructions;
        public readonly int HashCode;

        public KernelLookupKey(ShaderStage stage, byte flagsMask, ReadOnlySpan<Instruction> instructions, int hashCode)
        {
            Stage = stage;
            FlagsMask = flagsMask;
            Instructions = instructions;
            HashCode = hashCode;
        }
    }

    private sealed class KernelKeyComparer : IEqualityComparer<KernelCacheKey>, IAlternateEqualityComparer<KernelLookupKey, KernelCacheKey>
    {
        public static readonly KernelKeyComparer Instance = new();

        public bool Equals(KernelCacheKey x, KernelCacheKey y) => x.Equals(y);

        public int GetHashCode(KernelCacheKey obj) => obj.HashCode;

        public bool Equals(KernelLookupKey alternate, KernelCacheKey other)
        {
            return alternate.Stage == other.Stage
                && alternate.FlagsMask == other.FlagsMask
                && alternate.HashCode == other.HashCode
                && alternate.Instructions.SequenceEqual(other.Instructions);
        }

        public int GetHashCode(KernelLookupKey alternate) => alternate.HashCode;

        public KernelCacheKey Create(KernelLookupKey alternate)
        {
            return new KernelCacheKey(alternate.Stage, alternate.FlagsMask, alternate.Instructions.ToArray(), alternate.HashCode);
        }
    }
}

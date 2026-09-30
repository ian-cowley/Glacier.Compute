// <copyright file="AllocatorBenchmarks.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using Glacier.Compute;
using Glacier.Compute.Memory;

[MemoryDiagnoser]
public class AllocatorBenchmarks
{
    private SlabAllocator _slabAllocator = null!;
    private BuddyAllocator _buddyAllocator = null!;
    private RingScratchWorkspace _scratchWorkspace = null!;
    private DeviceMemoryPool _memoryPool = null!;

    [GlobalSetup]
    public void Setup()
    {
        _slabAllocator = new SlabAllocator(slotsPerBin: 1024);
        _buddyAllocator = new BuddyAllocator(64 * 1024 * 1024);
        _scratchWorkspace = new RingScratchWorkspace(64 * 1024 * 1024);
        _memoryPool = new DeviceMemoryPool();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _slabAllocator.Dispose();
        _buddyAllocator.Dispose();
        _scratchWorkspace.Dispose();
        _memoryPool.Dispose();
    }

    [Benchmark]
    public void SlabAllocateAndFree_1KB()
    {
        var ptr = _slabAllocator.Allocate(1024);
        _slabAllocator.Free(ptr);
    }

    [Benchmark]
    public void SlabAllocateAndFree_16KB()
    {
        var ptr = _slabAllocator.Allocate(16384);
        _slabAllocator.Free(ptr);
    }

    [Benchmark]
    public void BuddyAllocateAndFree_512KB()
    {
        var ptr = _buddyAllocator.Allocate(512 * 1024);
        _buddyAllocator.Free(ptr);
    }

    [Benchmark]
    public DevicePointer ScratchWorkspacePointerBump()
    {
        return _scratchWorkspace.Allocate(4096);
    }

    [Benchmark]
    public void PoolRentedScratchWorkspace()
    {
        using var ws = _memoryPool.RentScratchWorkspace(4096);
    }
}

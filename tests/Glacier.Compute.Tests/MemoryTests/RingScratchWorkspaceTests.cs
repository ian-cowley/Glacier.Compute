// <copyright file="RingScratchWorkspaceTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class RingScratchWorkspaceTests
{
    [Fact]
    public void RingScratchWorkspace_BumpsPointersAndWraps()
    {
        using var ring = new RingScratchWorkspace(256 * 1024);

        var p1 = ring.Allocate(64 * 1024);
        var p2 = ring.Allocate(64 * 1024);

        Assert.False(p1.IsNull);
        Assert.False(p2.IsNull);
        Assert.NotEqual(p1.Address, p2.Address);
        Assert.True(ring.Contains(p1.Address));
        Assert.True(ring.Contains(p2.Address));

        for (int i = 0; i < 64; i++)
        {
            var p = ring.Allocate(16 * 1024);
            Assert.False(p.IsNull);
            Assert.True(ring.Contains(p.Address));
            Assert.True(ring.Contains(p.Address + (nint)p.SizeInBytes - 1));
        }
    }

    [Fact]
    public void RingScratchWorkspace_ValidatesAlignedCapacityBounds()
    {
        // 96-byte ring: 64-byte aligned request fits, 65-byte request aligns to 128 > 96 and must throw
        using var ring = new RingScratchWorkspace(96);

        var ok = ring.Allocate(64);
        Assert.Equal(64UL, ok.SizeInBytes);

        Assert.Throws<ArgumentOutOfRangeException>(() => ring.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ring.Allocate(65));
        Assert.Throws<ArgumentOutOfRangeException>(() => ring.Allocate(128));
    }

    [Fact]
    public async Task RingScratchWorkspace_ConcurrentSinglePass_ProducesStrictlyNonOverlappingSlices()
    {
        const int threadCount = 8;
        const int allocsPerThread = 32;
        // Max total bytes = 8 * 32 * 512 = 131,072 bytes; use 256 KB ring so no wrap occurs
        using var ring = new RingScratchWorkspace(256 * 1024);
        IntPtr baseAddr = ring.BaseAddress;

        var slices = new ConcurrentBag<(long Start, long End)>();
        using var startGate = new ManualResetEventSlim(false);

        var tasks = Enumerable.Range(0, threadCount).Select(tid => Task.Run(() =>
        {
            startGate.Wait();
            for (int i = 0; i < allocsPerThread; i++)
            {
                ulong reqSize = (ulong)(((tid + i) % 8 + 1) * 64); // 64..512 bytes
                var ptr = ring.Allocate(reqSize);
                long relStart = ptr.Address - baseAddr;
                long relEnd = relStart + (long)ptr.SizeInBytes;
                slices.Add((relStart, relEnd));
            }
        })).ToArray();

        startGate.Set();
        await Task.WhenAll(tasks);

        var ordered = slices.OrderBy(s => s.Start).ToArray();
        Assert.Equal(threadCount * allocsPerThread, ordered.Length);

        for (int i = 0; i < ordered.Length; i++)
        {
            Assert.True(ordered[i].Start >= 0);
            Assert.True(ordered[i].End <= (long)ring.Capacity);
            Assert.Equal(0L, ordered[i].Start % 64);

            if (i > 0)
            {
                Assert.True(
                    ordered[i - 1].End <= ordered[i].Start,
                    $"Overlapping slices detected: [{ordered[i - 1].Start}, {ordered[i - 1].End}) and [{ordered[i].Start}, {ordered[i].End})");
            }
        }
    }

    [Fact]
    public async Task RingScratchWorkspace_ConcurrentWrapAround_NeverExceedsBoundsOrDesynchronizes()
    {
        // Small 4 KB ring with odd-multiple wrap sizes (e.g. 768 B, 960 B) to force frequent concurrent wrap padding
        const ulong capacity = 4096;
        using var ring = new RingScratchWorkspace(capacity);
        IntPtr baseAddr = ring.BaseAddress;

        const int threadCount = 8;
        const int iterations = 250;

        var tasks = Enumerable.Range(0, threadCount).Select(tid => Task.Run(() =>
        {
            for (int i = 0; i < iterations; i++)
            {
                ulong req = (ulong)(192 + ((tid + i) % 5) * 192); // 192, 384, 576, 768, 960 bytes
                var slice = ring.Allocate(req);
                long offset = slice.Address - baseAddr;

                Assert.True(offset >= 0, $"Negative ring offset {offset}");
                Assert.Equal(0L, offset % 64);
                Assert.True((ulong)offset + slice.SizeInBytes <= capacity, $"Slice [{offset}, {(ulong)offset + slice.SizeInBytes}) exceeded ring capacity {capacity}");
            }
        })).ToArray();

        await Task.WhenAll(tasks);
        Assert.True(ring.CurrentOffset < capacity);
    }

    [Fact]
    public void RingScratchWorkspace_ExternalArena_SupportsZeroCpuDereferenceAndReleaseCallback()
    {
        IntPtr externalBase = new IntPtr(0x0000_7100_0000_0000L);
        IntPtr released = IntPtr.Zero;

        using (var ring = new RingScratchWorkspace(externalBase, 8192, p => released = p))
        {
            var s1 = ring.Allocate(1000); // aligns to 1024
            var s2 = ring.Allocate(2000); // aligns to 2048
            Assert.Equal(externalBase, s1.Address);
            Assert.Equal(externalBase + 1024, s2.Address);
        }

        Assert.Equal(externalBase, released);
    }

    [Fact]
    public void DeviceMemoryPool_RentScratchWorkspace_ScopedDisposal()
    {
        using var pool = new DeviceMemoryPool(
            buddyArenaBytes: 1024 * 1024,
            virtualReserveBytes: 4 * 1024 * 1024,
            scratchBytes: 256 * 1024);

        IntPtr recordedAddress;
        using (var workspace = pool.RentScratchWorkspace(4096))
        {
            recordedAddress = workspace.Pointer;
            Assert.NotEqual(IntPtr.Zero, workspace.Pointer);
            Assert.True(workspace.Capacity >= 4096);
        }

        Assert.NotEqual(IntPtr.Zero, recordedAddress);
    }
}

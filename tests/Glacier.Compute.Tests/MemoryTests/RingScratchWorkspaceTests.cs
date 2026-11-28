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

        Assert.True(ring.Free(p1));
        Assert.True(ring.Free(p2));

        for (int i = 0; i < 64; i++)
        {
            var p = ring.Allocate(16 * 1024);
            Assert.False(p.IsNull);
            Assert.True(ring.Contains(p.Address));
            Assert.True(ring.Contains(p.Address + (nint)p.SizeInBytes - 1));
            Assert.True(ring.Free(p));
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
                var slice = ring.Allocate(req, TimeSpan.FromSeconds(5));
                long offset = slice.Address - baseAddr;

                Assert.True(offset >= 0, $"Negative ring offset {offset}");
                Assert.Equal(0L, offset % 64);
                Assert.True((ulong)offset + slice.SizeInBytes <= capacity, $"Slice [{offset}, {(ulong)offset + slice.SizeInBytes}) exceeded ring capacity {capacity}");
                Assert.True(ring.Free(slice));
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

    [Fact]
    public void RingScratchWorkspace_ActiveLeaseBlocksWraparoundOverwrite_AndOutOfOrderFreeUnblocks()
    {
        using var ring = new RingScratchWorkspace(256);

        var s1 = ring.Allocate(128);
        var s2 = ring.Allocate(128);

        // Ring is saturated: further allocation must not overwrite s1
        Assert.False(ring.TryAllocate(64, out var failed));
        Assert.True(failed.IsNull);
        Assert.Throws<OutOfMemoryException>(() => ring.Allocate(64));
        Assert.Throws<TimeoutException>(() => ring.Allocate(64, TimeSpan.FromMilliseconds(15)));

        // Out-of-order free of s2 keeps s1 blocking wraparound until s1 is also freed
        Assert.True(ring.Free(s2));
        Assert.False(ring.TryAllocate(64, out _));

        // Freeing s1 advances tail across both s1 and s2, unblocking full ring wraparound
        Assert.True(ring.Free(s1));
        Assert.False(ring.Free(s1)); // double free returns false

        var s3 = ring.Allocate(256);
        Assert.Equal(ring.BaseAddress, s3.Address);
        Assert.True(ring.Free(s3));
    }

    [Fact]
    public async Task RingScratchWorkspace_TimeoutGuards_AndInfiniteTimeoutRelease()
    {
        using var ring = new RingScratchWorkspace(128);
        Assert.Throws<ArgumentOutOfRangeException>(() => ring.TryAllocate(64, TimeSpan.FromMilliseconds(-5), out _));

        var held = ring.Allocate(128);

        var releaser = Task.Run(async () =>
        {
            await Task.Delay(30);
            ring.Free(held);
        });

        var next = ring.Allocate(128, Timeout.InfiniteTimeSpan);
        Assert.Equal(ring.BaseAddress, next.Address);
        Assert.True(ring.Free(next));
        await releaser;
    }

    [Fact]
    public unsafe void RingScratchWorkspace_WraparoundAfterPartialFree_AllowsLargerThanTailRemaining()
    {
        // Challenger Repro 3.4: 128 KB ring, 64 KB alloc + Free -> 96 KB alloc must succeed and be writable
        const ulong capacity = 128 * 1024;
        using var ring = new RingScratchWorkspace(capacity);

        var small = ring.Allocate(64 * 1024);
        Assert.Equal(ring.BaseAddress, small.Address);
        Assert.True(ring.Free(small));

        var large = ring.Allocate(96 * 1024);
        Assert.Equal(ring.BaseAddress, large.Address);
        Assert.Equal(96UL * 1024, large.SizeInBytes);

        var span = new Span<byte>((void*)large.Address, (int)large.SizeInBytes);
        span.Fill(0x5A);
        Assert.Equal((byte)0x5A, span[0]);
        Assert.Equal((byte)0x5A, span[^1]);

        Assert.True(ring.Free(large));
    }

    [Fact]
    public void RingScratchWorkspace_FullCapacityAllocation_SucceedsAfterArbitraryDrainAnd70kCycles()
    {
        // Arbitrary partial alloc+free cycles on 128 KB ring -> full 128 KB allocation when drained
        const ulong capacity = 128 * 1024;
        using (var ring = new RingScratchWorkspace(capacity))
        {
            ulong[] sizes = { 1024, 4096, 12288, 28672, 32768, 16384, 7680 };
            foreach (ulong size in sizes)
            {
                var slice = ring.Allocate(size);
                Assert.True(ring.Free(slice));

                var full = ring.Allocate(capacity);
                Assert.Equal(ring.BaseAddress, full.Address);
                Assert.Equal(capacity, full.SizeInBytes);
                Assert.True(ring.Free(full));
            }
        }

        // Challenger Repro 3.3: 70,000 sequential 64 B cycles -> full 4096 B allocation succeeds
        const ulong smallCap = 4096;
        using (var ring = new RingScratchWorkspace(smallCap))
        {
            for (int i = 0; i < 70_000; i++)
            {
                var s = ring.Allocate(64);
                Assert.True(ring.Free(s));
            }

            var full = ring.Allocate(smallCap);
            Assert.Equal(ring.BaseAddress, full.Address);
            Assert.Equal(smallCap, full.SizeInBytes);
            Assert.True(ring.Free(full));
        }
    }

    [Fact]
    public async Task RingScratchWorkspace_ConcurrentOutOfOrderStress_ZeroCorruptionAndFullCapacityRealloc()
    {
        // Challenger Repro 3.1: 8 threads x 2,000 out-of-order alloc/write-canary/verify/free -> full 256 KB alloc
        const ulong capacity = 256 * 1024;
        const int threadCount = 8;
        const int opsPerThread = 2000;
        using var ring = new RingScratchWorkspace(capacity);
        int corruptions = 0;

        var tasks = Enumerable.Range(0, threadCount).Select(tid => Task.Run(() =>
        {
            var rng = new Random(1000 + tid);
            var held = new System.Collections.Generic.List<(DevicePointer Ptr, byte Pattern)>(4);

            for (int i = 0; i < opsPerThread; i++)
            {
                ulong req = (ulong)(64 * (1 + rng.Next(16))); // 64..1024 bytes
                DevicePointer ptr;
                if (!ring.TryAllocate(req, out ptr))
                {
                    // Never block waiting for ring wraparound while holding our own older leases
                    // (a thread holding the tail lease would otherwise wait for itself).
                    while (held.Count > 0)
                    {
                        int idx = rng.Next(held.Count);
                        var (victim, expected) = held[idx];
                        held.RemoveAt(idx);
                        if (!VerifyCanary(victim, expected))
                        {
                            Interlocked.Increment(ref corruptions);
                        }

                        Assert.True(ring.Free(victim));
                        if (ring.TryAllocate(req, out ptr))
                        {
                            break;
                        }
                    }

                    if (ptr.IsNull)
                    {
                        ptr = ring.Allocate(req, TimeSpan.FromSeconds(10));
                    }
                }

                byte pattern = (byte)((tid * 31 + i) & 0xFF);
                WriteAndTrack(ptr, pattern);
                held.Add((ptr, pattern));

                if (held.Count >= 3 || (i == opsPerThread - 1))
                {
                    int releaseIdx = rng.Next(held.Count);
                    var (victim, expected) = held[releaseIdx];
                    held.RemoveAt(releaseIdx);
                    if (!VerifyCanary(victim, expected))
                    {
                        Interlocked.Increment(ref corruptions);
                    }

                    Assert.True(ring.Free(victim));
                }
            }

            foreach (var (rem, expected) in held)
            {
                if (!VerifyCanary(rem, expected))
                {
                    Interlocked.Increment(ref corruptions);
                }

                Assert.True(ring.Free(rem));
            }
        })).ToArray();

        await Task.WhenAll(tasks);
        Assert.Equal(0, corruptions);

        var full = ring.Allocate(capacity);
        Assert.Equal(ring.BaseAddress, full.Address);
        Assert.Equal(capacity, full.SizeInBytes);
        Assert.True(ring.Free(full));

        static unsafe void WriteAndTrack(DevicePointer ptr, byte pattern)
        {
            var span = new Span<byte>((void*)ptr.Address, (int)ptr.SizeInBytes);
            span.Fill(pattern);
        }

        static unsafe bool VerifyCanary(DevicePointer ptr, byte expected)
        {
            var span = new ReadOnlySpan<byte>((void*)ptr.Address, (int)ptr.SizeInBytes);
            for (int i = 0; i < span.Length; i++)
            {
                if (span[i] != expected)
                {
                    return false;
                }
            }

            return true;
        }
    }

    [Fact]
    public void RingScratchWorkspace_OutOfOrderFree_BlocksOverlappingWraparound_AllowsNonOverlappingWraparound()
    {
        const ulong capacity = 128 * 1024;
        using var ring = new RingScratchWorkspace(capacity);

        // Advance cursor to 64 KB and drain
        var warmup = ring.Allocate(64 * 1024);
        Assert.True(ring.Free(warmup));

        // Hold oldest lease at [64 KB, 96 KB) and a newer lease at [96 KB, 128 KB)
        var oldest = ring.Allocate(32 * 1024);
        Assert.Equal(ring.BaseAddress + (nint)(64 * 1024), oldest.Address);
        var newer = ring.Allocate(32 * 1024);
        Assert.Equal(ring.BaseAddress + (nint)(96 * 1024), newer.Address);

        // Free newer lease out-of-order while oldest lease at [64 KB, 96 KB) is still held
        Assert.True(ring.Free(newer));

        // Wraparound allocation of 96 KB ([0, 96 KB)) would overlap oldest [64 KB, 96 KB) -> must block
        Assert.False(ring.TryAllocate(96 * 1024, out _));

        // Non-overlapping wraparound of 64 KB ([0, 64 KB)) fits strictly before tailStart (64 KB) -> must succeed
        var wrapped = ring.Allocate(64 * 1024);
        Assert.Equal(ring.BaseAddress, wrapped.Address);

        // Now free oldest [64 KB, 96 KB) -> tailStart becomes 0, _writePos is 64 KB, freeing [64 KB, 128 KB)
        Assert.True(ring.Free(oldest));

        // 96 KB still cannot fit while wrapped [0, 64 KB) is held ([64 KB, 128 KB) is only 64 KB)
        Assert.False(ring.TryAllocate(96 * 1024, out _));

        // Free wrapped [0, 64 KB) -> ring drained, 96 KB now succeeds
        Assert.True(ring.Free(wrapped));
        var large = ring.Allocate(96 * 1024);
        Assert.Equal(96UL * 1024, large.SizeInBytes);
        Assert.True(ring.Free(large));
    }
}

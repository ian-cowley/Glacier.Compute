// <copyright file="VirtualMemoryReservoirTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using Glacier.Compute.Memory;
using Xunit;

public class VirtualMemoryReservoirTests
{
    [Fact]
    public void VirtualMemoryReservoir_CommitsAndFreesPages()
    {
        // 256 MB reservation
        using var reservoir = new VirtualMemoryReservoir(256 * 1024 * 1024);

        // Allocate 64 MB
        var p1 = reservoir.Allocate(64 * 1024 * 1024);
        Assert.False(p1.IsNull);
        Assert.True(p1.SizeInBytes >= 64 * 1024 * 1024);
        Assert.True(reservoir.Contains(p1.Address));

        // Allocate 32 MB
        var p2 = reservoir.Allocate(32 * 1024 * 1024);
        Assert.False(p2.IsNull);
        Assert.True(reservoir.Contains(p2.Address));

        Assert.True(reservoir.TotalCommittedBytes >= 96 * 1024 * 1024);

        // Free allocations
        Assert.True(reservoir.Free(p1));
        Assert.True(reservoir.Free(p2));
    }

    [Fact]
    public void VirtualMemoryReservoir_ThrowsWhenExhausted()
    {
        // 64 MB reservation
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        var p1 = reservoir.Allocate(64 * 1024 * 1024);
        Assert.False(p1.IsNull);

        // Exceeds total reservation
        Assert.Throws<OutOfMemoryException>(() => reservoir.Allocate(1024));
    }

    [Fact]
    public void VirtualMemoryReservoir_ReusesMiddleHole_BestFit_WithoutIncreasingCommittedWatermark()
    {
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        var a = reservoir.Allocate(8 * 1024 * 1024);
        var b = reservoir.Allocate(16 * 1024 * 1024);
        var c = reservoir.Allocate(8 * 1024 * 1024);
        ulong watermarkBeforeFree = reservoir.TotalCommittedBytes;

        Assert.True(reservoir.Free(b));
        Assert.Equal(16UL * 1024 * 1024, reservoir.ActiveAllocatedBytes);

        // Re-allocating 16 MB must reuse B's exact VA slot without advancing the committed watermark.
        var b2 = reservoir.Allocate(16 * 1024 * 1024);
        Assert.Equal(b.Address, b2.Address);
        Assert.Equal(watermarkBeforeFree, reservoir.TotalCommittedBytes);
        Assert.Equal(32UL * 1024 * 1024, reservoir.ActiveAllocatedBytes);

        Assert.True(reservoir.Free(a));
        Assert.True(reservoir.Free(b2));
        Assert.True(reservoir.Free(c));
        Assert.Equal(0UL, reservoir.ActiveAllocatedBytes);
    }

    [Fact]
    public void VirtualMemoryReservoir_CoalescesAdjacentHolesAndTailRewind()
    {
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        var p1 = reservoir.Allocate(16 * 1024 * 1024);
        var p2 = reservoir.Allocate(16 * 1024 * 1024);
        var p3 = reservoir.Allocate(16 * 1024 * 1024);
        var p4 = reservoir.Allocate(16 * 1024 * 1024);

        // Free middle blocks in reverse/non-adjacent order then merge across them
        Assert.True(reservoir.Free(p1));
        Assert.True(reservoir.Free(p3));
        Assert.True(reservoir.Free(p2)); // merges p1 + p2 + p3 into 48 MB hole

        var merged48 = reservoir.Allocate(48 * 1024 * 1024);
        Assert.Equal(p1.Address, merged48.Address);

        // Free tail p4 (and merged48), then verify full 64 MB can be allocated via tail rewind
        Assert.True(reservoir.Free(p4));
        Assert.True(reservoir.Free(merged48));

        var full64 = reservoir.Allocate(64 * 1024 * 1024);
        Assert.Equal(reservoir.BaseAddress, full64.Address);
        Assert.True(reservoir.Free(full64));
    }

    [Fact]
    public void VirtualMemoryReservoir_RepeatedAllocFreeCycles_NeverLeaksVirtualAddressSpace()
    {
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        // 1,000 x 8 MB = 8 GB cumulative allocations inside a 64 MB reservoir
        for (int i = 0; i < 1000; i++)
        {
            var ptr = reservoir.Allocate(8 * 1024 * 1024);
            Assert.True(reservoir.Contains(ptr.Address));
            Assert.True(reservoir.Free(ptr));
        }

        Assert.Equal(0UL, reservoir.ActiveAllocatedBytes);
    }

    [Fact]
    public void VirtualMemoryReservoir_ExternalArena_ReusesCoalescedIntervals()
    {
        IntPtr externalBase = new IntPtr(0x0000_7300_0000_0000L);
        using var reservoir = new VirtualMemoryReservoir(externalBase, 16UL * 1024 * 1024);

        var a = reservoir.Allocate(4 * 1024 * 1024);
        var b = reservoir.Allocate(4 * 1024 * 1024);
        var c = reservoir.Allocate(8 * 1024 * 1024);

        Assert.True(reservoir.Free(a));
        Assert.True(reservoir.Free(b));

        var ab = reservoir.Allocate(8 * 1024 * 1024);
        Assert.Equal(externalBase, ab.Address);
        Assert.True(reservoir.Free(ab));
        Assert.True(reservoir.Free(c));
        Assert.Equal(0UL, reservoir.ActiveAllocatedBytes);
    }
}

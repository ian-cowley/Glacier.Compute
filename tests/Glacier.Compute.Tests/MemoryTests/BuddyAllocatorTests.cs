// <copyright file="BuddyAllocatorTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class BuddyAllocatorTests
{
    [Fact]
    public void BuddyAllocator_CalculatesCorrectOrders()
    {
        Assert.Equal(0, BuddyAllocator.GetOrderForSize(128 * 1024));
        Assert.Equal(1, BuddyAllocator.GetOrderForSize(200 * 1024));
        Assert.Equal(1, BuddyAllocator.GetOrderForSize(256 * 1024));
        Assert.Equal(2, BuddyAllocator.GetOrderForSize(512 * 1024));
        Assert.Equal(3, BuddyAllocator.GetOrderForSize(1024 * 1024));
        Assert.Equal(8, BuddyAllocator.GetOrderForSize(32 * 1024 * 1024));
    }

    [Fact]
    public void BuddyAllocator_SplitsAndCoalescesBuddies()
    {
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);

        var p1 = allocator.Allocate(128 * 1024);
        var p2 = allocator.Allocate(128 * 1024);

        Assert.False(p1.IsNull);
        Assert.False(p2.IsNull);
        Assert.NotEqual(p1.Address, p2.Address);
        Assert.True(allocator.Contains(p1.Address));
        Assert.True(allocator.Contains(p2.Address));

        Assert.True(allocator.Free(p1));
        Assert.True(allocator.Free(p2));

        var pFull = allocator.Allocate(32 * 1024 * 1024);
        Assert.False(pFull.IsNull);
        Assert.Equal(32UL * 1024 * 1024, pFull.SizeInBytes);

        Assert.True(allocator.Free(pFull));
    }

    [Fact]
    public void BuddyAllocator_ExternalArena_SubAllocatesCoalescesAndReleases()
    {
        IntPtr externalBase = new IntPtr(0x0000_7300_0000_0000L);
        const ulong arenaBytes = 4UL * 1024 * 1024; // 4 MB
        IntPtr released = IntPtr.Zero;

        using (var allocator = new BuddyAllocator(externalBase, arenaBytes, p => released = p))
        {
            var a = allocator.Allocate(2 * 1024 * 1024);
            var b = allocator.Allocate(2 * 1024 * 1024);
            Assert.Equal(arenaBytes, allocator.AllocatedBytes);

            Assert.True(allocator.Free(a));
            Assert.True(allocator.Free(b));
            Assert.Equal(0UL, allocator.AllocatedBytes);

            // Coalesced back into full 4 MB block
            var merged = allocator.Allocate(4 * 1024 * 1024);
            Assert.Equal(externalBase, merged.Address);
            Assert.True(allocator.Free(merged));
        }

        Assert.Equal(externalBase, released);
    }

    [Fact]
    public void BuddyAllocator_ThrowsOnExcessiveSize()
    {
        using var allocator = new BuddyAllocator(1024 * 1024);
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(33 * 1024 * 1024));
    }
}

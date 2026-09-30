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
        Assert.Equal(0, BuddyAllocator.GetOrderForSize(128 * 1024)); // 128 KB -> Order 0
        Assert.Equal(1, BuddyAllocator.GetOrderForSize(200 * 1024)); // 200 KB -> Order 1 (256 KB)
        Assert.Equal(1, BuddyAllocator.GetOrderForSize(256 * 1024)); // 256 KB -> Order 1
        Assert.Equal(2, BuddyAllocator.GetOrderForSize(512 * 1024)); // 512 KB -> Order 2
        Assert.Equal(3, BuddyAllocator.GetOrderForSize(1024 * 1024)); // 1 MB -> Order 3
        Assert.Equal(8, BuddyAllocator.GetOrderForSize(32 * 1024 * 1024)); // 32 MB -> Order 8
    }

    [Fact]
    public void BuddyAllocator_SplitsAndCoalescesBuddies()
    {
        // 64 MB arena (two 32 MB blocks)
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);

        // Allocate two 128 KB blocks (will split root block down to order 0)
        var p1 = allocator.Allocate(128 * 1024);
        var p2 = allocator.Allocate(128 * 1024);

        Assert.False(p1.IsNull);
        Assert.False(p2.IsNull);
        Assert.NotEqual(p1.Address, p2.Address);
        Assert.True(allocator.Contains(p1.Address));
        Assert.True(allocator.Contains(p2.Address));

        // Free both blocks - they should recursively coalesce back together
        Assert.True(allocator.Free(p1));
        Assert.True(allocator.Free(p2));

        // Now allocate the entire 32 MB block; should succeed because coalescing restored it!
        var pFull = allocator.Allocate(32 * 1024 * 1024);
        Assert.False(pFull.IsNull);
        Assert.Equal(32UL * 1024 * 1024, pFull.SizeInBytes);

        Assert.True(allocator.Free(pFull));
    }

    [Fact]
    public void BuddyAllocator_ThrowsOnExcessiveSize()
    {
        using var allocator = new BuddyAllocator();
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(33 * 1024 * 1024));
    }
}

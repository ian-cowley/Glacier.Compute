// <copyright file="DeviceMemoryPoolTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.OrchestrationTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class DeviceMemoryPoolTests
{
    [Fact]
    public void DeviceMemoryPool_RoutesToCorrectTiers()
    {
        using var pool = new DeviceMemoryPool();

        // Small allocation <= 64 KB -> Slab
        var pSmall = pool.Allocate(1024, MemoryCategory.SlabSmall);
        Assert.False(pSmall.IsNull);
        Assert.True(pool.SlabAllocationsCount >= 1);
        Assert.True(pool.SlabTier.Contains(pSmall.Address));

        // Medium allocation <= 32 MB -> Buddy
        var pMed = pool.Allocate(512 * 1024, MemoryCategory.BuddyMedium);
        Assert.False(pMed.IsNull);
        Assert.True(pool.BuddyAllocationsCount >= 1);
        Assert.True(pool.BuddyTier.Contains(pMed.Address));

        // Large allocation > 32 MB -> Virtual
        var pLarge = pool.Allocate(40 * 1024 * 1024, MemoryCategory.LargeVirtual);
        Assert.False(pLarge.IsNull);
        Assert.True(pool.VirtualAllocationsCount >= 1);
        Assert.True(pool.VirtualTier.Contains(pLarge.Address));

        // Free each correctly
        pool.Free(pSmall);
        pool.Free(pMed);
        pool.Free(pLarge);
    }

    [Fact]
    public void DeviceMemoryPool_AutomaticRoutingBySize()
    {
        using var pool = new DeviceMemoryPool();

        var p1 = pool.Allocate(4096);
        Assert.True(pool.SlabTier.Contains(p1.Address));

        var p2 = pool.Allocate(1024 * 1024);
        Assert.True(pool.BuddyTier.Contains(p2.Address));

        pool.Free(p1);
        pool.Free(p2);
    }
}

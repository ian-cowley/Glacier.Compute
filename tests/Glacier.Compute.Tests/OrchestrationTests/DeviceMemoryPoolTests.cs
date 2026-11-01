// <copyright file="DeviceMemoryPoolTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.OrchestrationTests;

using System;
using System.Threading;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class DeviceMemoryPoolTests
{
    [Fact]
    public void DeviceMemoryPool_RoutesToCorrectTiers_WithCompactHostArenas()
    {
        // Compact arena sizes suitable for 2-vCPU CI runners
        using var pool = new DeviceMemoryPool(
            buddyArenaBytes: 4 * 1024 * 1024,
            virtualReserveBytes: 64 * 1024 * 1024,
            scratchBytes: 512 * 1024);

        // Small allocation <= 64 KB -> Slab
        var pSmall = pool.Allocate(1024, MemoryCategory.SlabSmall);
        Assert.False(pSmall.IsNull);
        Assert.True(pool.SlabAllocationsCount >= 1);
        Assert.True(pool.SlabTier.Contains(pSmall.Address));

        // Medium allocation <= buddy capacity -> Buddy
        var pMed = pool.Allocate(512 * 1024, MemoryCategory.BuddyMedium);
        Assert.False(pMed.IsNull);
        Assert.True(pool.BuddyAllocationsCount >= 1);
        Assert.True(pool.BuddyTier.Contains(pMed.Address));

        // Large allocation > 32 MB -> Virtual Reservoir
        var pLarge = pool.Allocate(40 * 1024 * 1024, MemoryCategory.LargeVirtual);
        Assert.False(pLarge.IsNull);
        Assert.True(pool.VirtualAllocationsCount >= 1);
        Assert.True(pool.VirtualTier.Contains(pLarge.Address));

        pool.Free(pSmall);
        pool.Free(pMed);
        pool.Free(pLarge);
    }

    [Fact]
    public void DeviceMemoryPool_AutomaticRoutingBySize()
    {
        using var pool = new DeviceMemoryPool(
            buddyArenaBytes: 4 * 1024 * 1024,
            virtualReserveBytes: 8 * 1024 * 1024,
            scratchBytes: 256 * 1024);

        var p1 = pool.Allocate(4096);
        Assert.True(pool.SlabTier.Contains(p1.Address));

        var p2 = pool.Allocate(1024 * 1024);
        Assert.True(pool.BuddyTier.Contains(p2.Address));

        pool.Free(p1);
        pool.Free(p2);
    }

    [Fact]
    public void DeviceMemoryPool_ExternalDeviceArena_SubAllocatesWithoutCpuDereference_AndInvokesReleaseAction()
    {
        // Synthetic unmapped GPU device address (proves CPU never dereferences external VRAM pointers)
        IntPtr syntheticDeviceVramBase = new IntPtr(0x0000_7000_0000_0000L);
        const ulong arenaSize = 16UL * 1024 * 1024; // 16 MB
        int releaseCalls = 0;
        IntPtr releasedPtr = IntPtr.Zero;

        using (var pool = new DeviceMemoryPool(
            syntheticDeviceVramBase,
            arenaSize,
            ptr =>
            {
                Interlocked.Increment(ref releaseCalls);
                releasedPtr = ptr;
            }))
        {
            var slabPtr = pool.Allocate(2048, MemoryCategory.SlabSmall);
            var buddyPtr = pool.Allocate(512 * 1024, MemoryCategory.BuddyMedium);
            var largePtr = pool.Allocate(1024 * 1024, MemoryCategory.LargeVirtual);

            Assert.InRange((long)slabPtr.Address, (long)syntheticDeviceVramBase, (long)syntheticDeviceVramBase + (long)arenaSize - 1);
            Assert.InRange((long)buddyPtr.Address, (long)syntheticDeviceVramBase, (long)syntheticDeviceVramBase + (long)arenaSize - 1);
            Assert.InRange((long)largePtr.Address, (long)syntheticDeviceVramBase, (long)syntheticDeviceVramBase + (long)arenaSize - 1);

            using (var scratch = pool.RentScratchWorkspace(4096))
            {
                Assert.InRange((long)scratch.Pointer, (long)syntheticDeviceVramBase, (long)syntheticDeviceVramBase + (long)arenaSize - 1);
            }

            pool.Free(slabPtr);
            pool.Free(buddyPtr);
            pool.Free(largePtr);
        }

        Assert.Equal(1, releaseCalls);
        Assert.Equal(syntheticDeviceVramBase, releasedPtr);
    }
}

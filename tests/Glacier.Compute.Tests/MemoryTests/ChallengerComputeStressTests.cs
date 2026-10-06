// <copyright file="ChallengerComputeStressTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class ChallengerComputeStressTests
{
    [Fact]
    public void BuddyAllocator_EdgeConditions_ZeroBytes_OOM_DoubleFree_ForeignPointer()
    {
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);

        // 1. Zero-byte allocation
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));

        // 2. Allocation larger than max block size (32 MB)
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(33 * 1024 * 1024));

        // 3. Exhaust arena (64 MB = two 32 MB root blocks)
        var p1 = allocator.Allocate(32 * 1024 * 1024);
        var p2 = allocator.Allocate(32 * 1024 * 1024);
        Assert.Equal(64UL * 1024 * 1024, allocator.AllocatedBytes);

        // 4. OOM exception throwing when arena is full
        var oomEx = Assert.Throws<OutOfMemoryException>(() => allocator.Allocate(128 * 1024));
        Assert.Contains("BuddyAllocator arena exhausted", oomEx.Message);

        // 5. Foreign pointer frees
        Assert.False(allocator.Free(DevicePointer.Null));
        Assert.False(allocator.Free(new DevicePointer((IntPtr)0x12345678, 128 * 1024)));
        Assert.False(allocator.Free(new DevicePointer(IntPtr.Zero, 128 * 1024)));

        // Unaligned / interior pointer within arena but not at block boundary
        IntPtr interiorPtr = p1.Address + 4096;
        Assert.True(allocator.Contains(interiorPtr));
        Assert.False(allocator.Free(new DevicePointer(interiorPtr, 128 * 1024)));

        // 6. Legitimate free
        Assert.True(allocator.Free(p1));
        Assert.Equal(32UL * 1024 * 1024, allocator.AllocatedBytes);

        // 7. Double free on p1
        Assert.False(allocator.Free(p1));
        Assert.Equal(32UL * 1024 * 1024, allocator.AllocatedBytes);

        // 8. Re-allocate into freed buddy space
        var p3 = allocator.Allocate(16 * 1024 * 1024);
        var p4 = allocator.Allocate(16 * 1024 * 1024);
        Assert.Equal(64UL * 1024 * 1024, allocator.AllocatedBytes);

        // Free all and verify complete coalescing back to 0
        Assert.True(allocator.Free(p2));
        Assert.True(allocator.Free(p3));
        Assert.True(allocator.Free(p4));
        Assert.Equal(0UL, allocator.AllocatedBytes);

        // After coalescing, we should be able to allocate a full 32 MB block again
        var pRoot = allocator.Allocate(32 * 1024 * 1024);
        Assert.True(allocator.Free(pRoot));
    }

    [Fact]
    public void BuddyAllocator_CoalescingStressTest_FragmentAndMerge()
    {
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);
        var pointers = new List<DevicePointer>();

        // Allocate 64 x 128 KB blocks (8 MB total)
        for (int i = 0; i < 64; i++)
        {
            pointers.Add(allocator.Allocate(128 * 1024));
        }
        Assert.Equal(64UL * 128 * 1024, allocator.AllocatedBytes);

        // Free in odd/even alternating order to stress recursive coalescing
        for (int i = 1; i < pointers.Count; i += 2)
        {
            Assert.True(allocator.Free(pointers[i]));
        }
        for (int i = 0; i < pointers.Count; i += 2)
        {
            Assert.True(allocator.Free(pointers[i]));
        }

        Assert.Equal(0UL, allocator.AllocatedBytes);

        // Allocate root 32 MB blocks to verify full coalescing was successful
        var p1 = allocator.Allocate(32 * 1024 * 1024);
        var p2 = allocator.Allocate(32 * 1024 * 1024);
        Assert.True(allocator.Free(p1));
        Assert.True(allocator.Free(p2));
        Assert.Equal(0UL, allocator.AllocatedBytes);
    }

    [Fact]
    public async Task BuddyAllocator_ConcurrentMultiThreadedAllocFree_NoCorruption()
    {
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);
        const int threadCount = 6;
        const int iterations = 50;

        var tasks = Enumerable.Range(0, threadCount).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < iterations; i++)
            {
                DevicePointer ptr;
                try
                {
                    ptr = allocator.Allocate(128 * 1024);
                }
                catch (OutOfMemoryException)
                {
                    // Arena temporarily exhausted under concurrency
                    continue;
                }

                Assert.True(allocator.Contains(ptr.Address));
                bool freed = allocator.Free(ptr);
                Assert.True(freed);
            }
        })).ToArray();

        await Task.WhenAll(tasks);
        Assert.Equal(0UL, allocator.AllocatedBytes);
    }

    [Fact]
    public void SlabAllocator_EdgeConditions_ZeroBytes_OverMax_OOM_Misaligned_DoubleFree()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 64);

        // 1. Zero-byte allocation
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));

        // 2. Over max slab size (64 KB)
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(65537));

        // 3. Foreign pointer frees
        Assert.False(allocator.Free(DevicePointer.Null));
        Assert.False(allocator.Free(new DevicePointer((IntPtr)0x12345678, 1024)));

        // 4. Exhaust 256 B bin (64 slots)
        var allocated = new List<DevicePointer>();
        for (int i = 0; i < 64; i++)
        {
            allocated.Add(allocator.Allocate(256));
        }

        // Try 65th allocation in 256 B bin: it should fall through to next larger bin (512 B)
        var fallbackPtr = allocator.Allocate(256);
        Assert.Equal(512UL, fallbackPtr.SizeInBytes);

        // 5. Misaligned free inside bin
        var first = allocated[0];
        Assert.False(allocator.Free(new DevicePointer(first.Address + 1, 256)));
        Assert.False(allocator.Free(new DevicePointer(first.Address + 3, 256)));

        // 6. Free and double free
        Assert.True(allocator.Free(first));
        Assert.False(allocator.Free(first));

        // 7. Cleanup remaining
        Assert.True(allocator.Free(fallbackPtr));
        for (int i = 1; i < allocated.Count; i++)
        {
            Assert.True(allocator.Free(allocated[i]));
        }
        Assert.Equal(0UL, allocator.ActiveAllocatedBytes);
    }

    [Fact]
    public async Task SlabAllocator_ConcurrentMultiThreadedAllocFree_NoCollisions()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 1024);
        const int threadCount = 8;
        const int iterations = 100;

        var tasks = Enumerable.Range(0, threadCount).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < iterations; i++)
            {
                var ptr = allocator.Allocate(512);
                Assert.True(allocator.Contains(ptr.Address));
                bool freed = allocator.Free(ptr);
                Assert.True(freed);
            }
        })).ToArray();

        await Task.WhenAll(tasks);
        Assert.Equal(0UL, allocator.ActiveAllocatedBytes);
    }

    [Fact]
    public void VirtualMemoryReservoir_EdgeConditions_ZeroBytes_OOM_DoubleFree_Foreign()
    {
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        // 1. Zero-byte allocation
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(0));

        // 2. Over capacity allocation
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(65 * 1024 * 1024));

        // 3. Foreign pointer frees
        Assert.False(reservoir.Free(DevicePointer.Null));
        Assert.False(reservoir.Free(new DevicePointer((IntPtr)0x12345678, 1024 * 1024)));

        // 4. Exhaust reservoir
        var p1 = reservoir.Allocate(32 * 1024 * 1024);
        var p2 = reservoir.Allocate(32 * 1024 * 1024);
        Assert.Equal(64UL * 1024 * 1024, reservoir.TotalCommittedBytes);

        // 5. OOM when exhausted
        var ex = Assert.Throws<OutOfMemoryException>(() => reservoir.Allocate(65536));
        Assert.Contains("Virtual memory reservoir exhausted", ex.Message);

        // 6. Free and double free
        Assert.True(reservoir.Free(p1));
        Assert.False(reservoir.Free(p1)); // Double free returns false

        // Unaligned / interior pointer free returns false
        Assert.False(reservoir.Free(new DevicePointer(p2.Address + 4096, 65536)));

        Assert.True(reservoir.Free(p2));
        Assert.Equal(0UL, reservoir.ActiveAllocatedBytes);
    }
}

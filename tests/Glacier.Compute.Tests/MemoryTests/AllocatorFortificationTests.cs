// <copyright file="AllocatorFortificationTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using System.Runtime.InteropServices;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class AllocatorFortificationTests
{
    [Fact]
    public void BuddyAllocator_DisposalGuards_ThrowObjectDisposedException()
    {
        var allocator = new BuddyAllocator(64 * 1024 * 1024);
        var ptr = allocator.Allocate(128 * 1024);
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() => allocator.Allocate(128 * 1024));
        Assert.Throws<ObjectDisposedException>(() => allocator.Free(ptr));
        Assert.Throws<ObjectDisposedException>(() => allocator.Contains(ptr.Address));
        Assert.Throws<ObjectDisposedException>(() => _ = allocator.TotalCapacity);
        Assert.Throws<ObjectDisposedException>(() => _ = allocator.AllocatedBytes);

        // Multiple dispose calls must be completely idempotent and safe
        allocator.Dispose();
        allocator.Dispose();
    }

    [Fact]
    public void BuddyAllocator_ForeignAndInvalidPointers_HandledSafely()
    {
        using var allocator = new BuddyAllocator(64 * 1024 * 1024);

        // Null pointer free returns false
        Assert.False(allocator.Free(DevicePointer.Null));

        // Foreign pointer outside arena returns false
        IntPtr foreignAddr = (IntPtr)0x12345678;
        Assert.False(allocator.Contains(foreignAddr));
        Assert.False(allocator.Free(new DevicePointer(foreignAddr, 128 * 1024)));

        // Zero address check
        Assert.False(allocator.Contains(IntPtr.Zero));

        // Legitimate allocation
        var p = allocator.Allocate(128 * 1024);
        Assert.True(allocator.Contains(p.Address));
        Assert.Equal(128UL * 1024, allocator.AllocatedBytes);

        // Free succeeds
        Assert.True(allocator.Free(p));
        Assert.Equal(0UL, allocator.AllocatedBytes);

        // Double free returns false
        Assert.False(allocator.Free(p));
    }

    [Fact]
    public void BuddyAllocator_BoundsAndDiagnostics_Validated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BuddyAllocator.GetOrderForSize(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => BuddyAllocator.GetBlockSize(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => BuddyAllocator.GetBlockSize(BuddyAllocator.OrderCount));

        using var allocator = new BuddyAllocator(64 * 1024 * 1024);

        // Allocate both 32 MB root blocks
        var p1 = allocator.Allocate(32 * 1024 * 1024);
        var p2 = allocator.Allocate(32 * 1024 * 1024);
        Assert.Equal(64UL * 1024 * 1024, allocator.AllocatedBytes);

        // Third allocation must fail with informative OutOfMemoryException
        var ex = Assert.Throws<OutOfMemoryException>(() => allocator.Allocate(128 * 1024));
        Assert.Contains("BuddyAllocator arena exhausted", ex.Message);
        Assert.Contains("Total capacity", ex.Message);
        Assert.Contains("Allocated", ex.Message);

        allocator.Free(p1);
        allocator.Free(p2);
        Assert.Equal(0UL, allocator.AllocatedBytes);
    }

    [Fact]
    public void SlabAllocator_DisposalGuards_ThrowObjectDisposedException()
    {
        var allocator = new SlabAllocator(slotsPerBin: 64);
        var ptr = allocator.Allocate(1024);
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() => allocator.Allocate(1024));
        Assert.Throws<ObjectDisposedException>(() => allocator.Free(ptr));
        Assert.Throws<ObjectDisposedException>(() => allocator.Contains(ptr.Address));
        Assert.Throws<ObjectDisposedException>(() => _ = allocator.TotalCommittedBytes);
        Assert.Throws<ObjectDisposedException>(() => _ = allocator.ActiveAllocatedBytes);

        // Idempotent dispose
        allocator.Dispose();
        allocator.Dispose();
    }

    [Fact]
    public void SlabAllocator_ForeignAndMisalignedPointers_HandledSafely()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 64);

        // Null pointer free returns false
        Assert.False(allocator.Free(DevicePointer.Null));

        // Foreign pointer returns false
        IntPtr foreignAddr = (IntPtr)0x12345678;
        Assert.False(allocator.Contains(foreignAddr));
        Assert.False(allocator.Free(new DevicePointer(foreignAddr, 1024)));
        Assert.False(allocator.Contains(IntPtr.Zero));

        // Valid allocation
        var ptr = allocator.Allocate(1024);
        Assert.True(allocator.Contains(ptr.Address));
        Assert.True(allocator.ActiveAllocatedBytes >= 1024);

        // Misaligned pointer within the slab bin returns false
        var misalignedPtr = new DevicePointer(ptr.Address + 1, 1024);
        Assert.False(allocator.Free(misalignedPtr));

        // Legitimate free
        Assert.True(allocator.Free(ptr));

        // Double free returns false
        Assert.False(allocator.Free(ptr));
    }

    [Fact]
    public void SlabAllocator_BoundsAndDiagnostics_Validated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlabAllocator(slotsPerBin: int.MaxValue));

        using var allocator = new SlabAllocator(slotsPerBin: 64);
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(70000));

        // Exhaust the 65536 bin (64 slots)
        var list = new System.Collections.Generic.List<DevicePointer>();
        for (int i = 0; i < 64; i++)
        {
            list.Add(allocator.Allocate(65536));
        }

        var ex = Assert.Throws<OutOfMemoryException>(() => allocator.Allocate(65536));
        Assert.Contains("SlabAllocator exhausted", ex.Message);
        Assert.Contains("Active allocated", ex.Message);
        Assert.Contains("Total committed", ex.Message);

        foreach (var p in list)
        {
            allocator.Free(p);
        }
        Assert.Equal(0UL, allocator.ActiveAllocatedBytes);
    }

    [Fact]
    public void VirtualMemoryReservoir_DisposalGuards_ThrowObjectDisposedException()
    {
        var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);
        var ptr = reservoir.Allocate(1024 * 1024);
        reservoir.Dispose();

        Assert.Throws<ObjectDisposedException>(() => reservoir.Allocate(1024 * 1024));
        Assert.Throws<ObjectDisposedException>(() => reservoir.Free(ptr));
        Assert.Throws<ObjectDisposedException>(() => reservoir.Contains(ptr.Address));
        Assert.Throws<ObjectDisposedException>(() => _ = reservoir.TotalReservedBytes);
        Assert.Throws<ObjectDisposedException>(() => _ = reservoir.TotalCommittedBytes);
        Assert.Throws<ObjectDisposedException>(() => _ = reservoir.ActiveAllocatedBytes);

        // Idempotent dispose
        reservoir.Dispose();
        reservoir.Dispose();
    }

    [Fact]
    public void VirtualMemoryReservoir_IntegerOverflowAndBoundsGuards()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new VirtualMemoryReservoir(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VirtualMemoryReservoir(ulong.MaxValue));

        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        // Allocation size 0
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(0));

        // Allocation size exceeds reserved capacity
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(65 * 1024 * 1024));

        // Integer overflow attack values
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(ulong.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(ulong.MaxValue - 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => reservoir.Allocate(ulong.MaxValue - 65535UL));
    }

    [Fact]
    public void VirtualMemoryReservoir_ForeignPointersAndDiagnostics()
    {
        using var reservoir = new VirtualMemoryReservoir(64 * 1024 * 1024);

        Assert.False(reservoir.Free(DevicePointer.Null));
        Assert.False(reservoir.Contains(IntPtr.Zero));

        IntPtr foreignAddr = (IntPtr)0x12345678;
        Assert.False(reservoir.Contains(foreignAddr));
        Assert.False(reservoir.Free(new DevicePointer(foreignAddr, 1024 * 1024)));

        var p = reservoir.Allocate(2 * 1024 * 1024);
        Assert.True(reservoir.Contains(p.Address));
        Assert.True(reservoir.ActiveAllocatedBytes >= 2 * 1024 * 1024);

        Assert.True(reservoir.Free(p));
        Assert.Equal(0UL, reservoir.ActiveAllocatedBytes);

        // Double free returns false
        Assert.False(reservoir.Free(p));
    }
}

// <copyright file="SlabAllocatorTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class SlabAllocatorTests
{
    [Fact]
    public void SlabAllocator_LazilyMaterializesBinsOnFirstDemand()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 64);

        // Zero committed host bytes prior to first allocation
        Assert.Equal(0UL, allocator.TotalCommittedBytes);

        // Touch only the 1024-byte bin -> only 64 * 1024 = 65,536 bytes committed
        var ptr = allocator.Allocate(1024);
        Assert.Equal(64UL * 1024UL, allocator.TotalCommittedBytes);
        Assert.True(allocator.Free(ptr));
    }

    [Fact]
    public void SlabAllocator_AllocatesAndFreesAllBinSizes()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 64);

        foreach (int binSize in SlabAllocator.BinSizes)
        {
            var ptr = allocator.Allocate((ulong)binSize);
            Assert.False(ptr.IsNull);
            Assert.True(ptr.SizeInBytes >= (ulong)binSize);
            Assert.True(allocator.Contains(ptr.Address));

            bool freed = allocator.Free(ptr);
            Assert.True(freed);
        }
    }

    [Fact]
    public void SlabAllocator_MassiveAllocationAndFreeCycle_Succeeds()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 128);
        const int count = 64;
        var pointers = new DevicePointer[count];

        for (int i = 0; i < count; i++)
        {
            pointers[i] = allocator.Allocate(1024);
            Assert.False(pointers[i].IsNull);
            Assert.True(allocator.Contains(pointers[i].Address));
        }

        Assert.True(allocator.ActiveAllocatedBytes >= count * 1024);

        for (int i = 0; i < count; i++)
        {
            Assert.True(allocator.Free(pointers[i]));
        }

        for (int i = 0; i < count; i++)
        {
            Assert.False(allocator.Free(pointers[i]));
        }
    }

    [Fact]
    public void SlabAllocator_ExternalArena_SubAllocatesAndReleases()
    {
        IntPtr externalBase = new IntPtr(0x0000_7200_0000_0000L);
        const ulong byteLength = 1024 * 1024; // 1 MB -> 8 slots per bin across all 9 bins
        IntPtr released = IntPtr.Zero;

        using (var allocator = new SlabAllocator(externalBase, byteLength, p => released = p))
        {
            var p256 = allocator.Allocate(256);
            var p64k = allocator.Allocate(65536);

            Assert.True(allocator.Contains(p256.Address));
            Assert.True(allocator.Contains(p64k.Address));
            Assert.True(allocator.Free(p256));
            Assert.True(allocator.Free(p64k));
        }

        Assert.Equal(externalBase, released);
    }

    [Fact]
    public void SlabAllocator_ThrowsOnInvalidSize()
    {
        using var allocator = new SlabAllocator(slotsPerBin: 64);

        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(65537));
    }
}

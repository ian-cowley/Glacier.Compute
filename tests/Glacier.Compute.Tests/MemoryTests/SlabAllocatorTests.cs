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

        // Second free attempt should fail (double free prevention)
        for (int i = 0; i < count; i++)
        {
            Assert.False(allocator.Free(pointers[i]));
        }
    }

    [Fact]
    public void SlabAllocator_ThrowsOnInvalidSize()
    {
        using var allocator = new SlabAllocator();

        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(65537));
    }
}

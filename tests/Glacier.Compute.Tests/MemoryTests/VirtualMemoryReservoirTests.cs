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
}

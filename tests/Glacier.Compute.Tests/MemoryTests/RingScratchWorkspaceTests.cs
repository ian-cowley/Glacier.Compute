// <copyright file="RingScratchWorkspaceTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.MemoryTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Memory;
using Xunit;

public class RingScratchWorkspaceTests
{
    [Fact]
    public void RingScratchWorkspace_BumpsPointersAndWraps()
    {
        // 1 MB ring buffer
        using var ring = new RingScratchWorkspace(1024 * 1024);

        var p1 = ring.Allocate(256 * 1024);
        var p2 = ring.Allocate(256 * 1024);

        Assert.False(p1.IsNull);
        Assert.False(p2.IsNull);
        Assert.NotEqual(p1.Address, p2.Address);
        Assert.True(ring.Contains(p1.Address));
        Assert.True(ring.Contains(p2.Address));

        // Rapid bump allocations across multiple loops
        for (int i = 0; i < 100; i++)
        {
            var p = ring.Allocate(64 * 1024);
            Assert.False(p.IsNull);
            Assert.True(ring.Contains(p.Address));
        }
    }

    [Fact]
    public void DeviceMemoryPool_RentScratchWorkspace_ScopedDisposal()
    {
        using var pool = new DeviceMemoryPool();

        IntPtr recordedAddress;
        using (var workspace = pool.RentScratchWorkspace(4096))
        {
            recordedAddress = workspace.Pointer;
            Assert.NotEqual(IntPtr.Zero, workspace.Pointer);
            Assert.True(workspace.Capacity >= 4096);
        }

        // Disposal should not crash
        Assert.NotEqual(IntPtr.Zero, recordedAddress);
    }
}

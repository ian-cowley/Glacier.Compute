// <copyright file="DeviceMemoryPool.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Threading;

/// <summary>
/// Unified multi-tier VRAM memory pool orchestrator implementing <see cref="IVramAllocator"/>.
/// Coordinates micro-slabs (256B–64KB), buddy power-of-two blocks (128KB–32MB),
/// virtual memory page reservations (&gt; 32MB), and stream scratch workspaces.
/// </summary>
public sealed class DeviceMemoryPool : IVramAllocator
{
    private readonly SlabAllocator _slabAllocator;
    private readonly BuddyAllocator _buddyAllocator;
    private readonly VirtualMemoryReservoir _virtualReservoir;
    private readonly RingScratchWorkspace _scratchWorkspace;

    private long _slabAllocations;
    private long _buddyAllocations;
    private long _virtualAllocations;
    private bool _disposed;

    /// <summary>Gets the micro-slab tier allocator.</summary>
    public SlabAllocator SlabTier => _slabAllocator;

    /// <summary>Gets the buddy power-of-two tier allocator.</summary>
    public BuddyAllocator BuddyTier => _buddyAllocator;

    /// <summary>Gets the virtual address page reservoir tier.</summary>
    public VirtualMemoryReservoir VirtualTier => _virtualReservoir;

    /// <summary>Gets the lock-free stream ring scratch workspace.</summary>
    public RingScratchWorkspace ScratchTier => _scratchWorkspace;

    /// <summary>Gets the count of micro-slab allocations serviced.</summary>
    public long SlabAllocationsCount => Volatile.Read(ref _slabAllocations);

    /// <summary>Gets the count of buddy block allocations serviced.</summary>
    public long BuddyAllocationsCount => Volatile.Read(ref _buddyAllocations);

    /// <summary>Gets the count of virtual reservoir allocations serviced.</summary>
    public long VirtualAllocationsCount => Volatile.Read(ref _virtualAllocations);

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceMemoryPool"/> class.
    /// </summary>
    /// <param name="buddyArenaBytes">Size of the buddy allocator arena (default 64 MB).</param>
    /// <param name="virtualReserveBytes">Size of the virtual page reservoir (default 4 GB).</param>
    /// <param name="scratchBytes">Size of the per-stream circular scratch buffer (default 64 MB).</param>
    public DeviceMemoryPool(
        ulong buddyArenaBytes = BuddyAllocator.MaxBlockSize * 2,
        ulong virtualReserveBytes = 4UL * 1024 * 1024 * 1024,
        ulong scratchBytes = 64UL * 1024 * 1024)
    {
        _slabAllocator = new SlabAllocator();
        _buddyAllocator = new BuddyAllocator(buddyArenaBytes);
        _virtualReservoir = new VirtualMemoryReservoir(virtualReserveBytes);
        _scratchWorkspace = new RingScratchWorkspace(scratchBytes);
    }

    /// <inheritdoc/>
    public DevicePointer Allocate(ulong byteCount, MemoryCategory category)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (byteCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Allocation size must be greater than zero.");
        }

        // Intelligently route based on explicit category or size threshold
        switch (category)
        {
            case MemoryCategory.SlabSmall:
                if (byteCount <= 65536)
                {
                    Interlocked.Increment(ref _slabAllocations);
                    return _slabAllocator.Allocate(byteCount);
                }
                goto case MemoryCategory.BuddyMedium;

            case MemoryCategory.BuddyMedium:
                if (byteCount <= BuddyAllocator.MaxBlockSize)
                {
                    Interlocked.Increment(ref _buddyAllocations);
                    return _buddyAllocator.Allocate(byteCount);
                }
                goto case MemoryCategory.LargeVirtual;

            case MemoryCategory.LargeVirtual:
            default:
                Interlocked.Increment(ref _virtualAllocations);
                return _virtualReservoir.Allocate(byteCount);
        }
    }

    /// <summary>
    /// Automatically routes allocation to the optimal memory tier based solely on byte size.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        if (byteCount <= 65536)
        {
            return Allocate(byteCount, MemoryCategory.SlabSmall);
        }

        if (byteCount <= BuddyAllocator.MaxBlockSize)
        {
            return Allocate(byteCount, MemoryCategory.BuddyMedium);
        }

        return Allocate(byteCount, MemoryCategory.LargeVirtual);
    }

    /// <inheritdoc/>
    public void Free(DevicePointer ptr)
    {
        if (_disposed || ptr.IsNull)
        {
            return;
        }

        // Fast path check based on address containment
        if (_slabAllocator.Contains(ptr.Address))
        {
            _slabAllocator.Free(ptr);
            return;
        }

        if (_buddyAllocator.Contains(ptr.Address))
        {
            _buddyAllocator.Free(ptr);
            return;
        }

        if (_virtualReservoir.Contains(ptr.Address))
        {
            _virtualReservoir.Free(ptr);
            return;
        }

        // Scratch workspaces are managed via circular head bumping, no explicit free needed
    }

    /// <inheritdoc/>
    public ScratchWorkspace RentScratchWorkspace(ulong minimumBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ptr = _scratchWorkspace.Allocate(minimumBytes);
        return new ScratchWorkspace(ptr.Address, ptr.SizeInBytes, this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _slabAllocator.Dispose();
            _buddyAllocator.Dispose();
            _virtualReservoir.Dispose();
            _scratchWorkspace.Dispose();
        }
    }
}

// <copyright file="DeviceMemoryPool.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Threading;

/// <summary>
/// Unified multi-tier memory pool orchestrator implementing <see cref="IVramAllocator"/>.
/// Coordinates micro-slabs (256 B – 64 KB), buddy power-of-two blocks (128 KB – 32 MB),
/// large reservoirs (&gt; 32 MB), and circular stream scratch workspaces.
/// Can sub-allocate caller-supplied GPU VRAM / pinned arenas (<c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>D3D12</c>)
/// or standalone unmanaged host memory arenas when constructed via the default host constructor.
/// </summary>
public sealed class DeviceMemoryPool : IVramAllocator
{
    /// <summary>Minimum byte length required to partition a single external arena across all four memory tiers (1 MB).</summary>
    public const ulong MinimumExternalArenaBytes = 1024UL * 1024UL;

    private readonly SlabAllocator _slabAllocator;
    private readonly BuddyAllocator _buddyAllocator;
    private readonly VirtualMemoryReservoir _virtualReservoir;
    private readonly RingScratchWorkspace _scratchWorkspace;
    private IntPtr _externalBase;
    private readonly Action<IntPtr>? _releaseAction;

    private long _slabAllocations;
    private long _buddyAllocations;
    private long _virtualAllocations;
    private int _disposed;

    /// <summary>Gets the micro-slab tier allocator.</summary>
    public SlabAllocator SlabTier => _slabAllocator;

    /// <summary>Gets the buddy power-of-two tier allocator.</summary>
    public BuddyAllocator BuddyTier => _buddyAllocator;

    /// <summary>Gets the large reservoir tier.</summary>
    public VirtualMemoryReservoir VirtualTier => _virtualReservoir;

    /// <summary>Gets the lock-free stream ring scratch workspace.</summary>
    public RingScratchWorkspace ScratchTier => _scratchWorkspace;

    /// <summary>Gets the count of micro-slab allocations serviced.</summary>
    public long SlabAllocationsCount => Volatile.Read(ref _slabAllocations);

    /// <summary>Gets the count of buddy block allocations serviced.</summary>
    public long BuddyAllocationsCount => Volatile.Read(ref _buddyAllocations);

    /// <summary>Gets the count of large reservoir allocations serviced.</summary>
    public long VirtualAllocationsCount => Volatile.Read(ref _virtualAllocations);

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceMemoryPool"/> class backed by standalone unmanaged host memory arenas
    /// (lazy slab bins, aligned host buddy/scratch buffers, and zero-RSS OS virtual address space reservation).
    /// To sub-allocate real GPU VRAM or pinned device memory, use <see cref="DeviceMemoryPool(IntPtr, ulong, Action{IntPtr}?)"/>.
    /// </summary>
    /// <param name="buddyArenaBytes">Size of the host buddy allocator arena (default 64 MB).</param>
    /// <param name="virtualReserveBytes">Size of the zero-RSS host virtual page reservoir (default 4 GB).</param>
    /// <param name="scratchBytes">Size of the host circular scratch buffer (default 64 MB).</param>
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

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceMemoryPool"/> class that sub-allocates an external GPU VRAM
    /// or pinned memory arena (e.g. CUDA <c>cuMemAlloc</c> / <c>cuMemHostAlloc</c>, Direct3D 12 heap, or Vulkan device memory).
    /// Partitions <paramref name="baseAddress"/> into 64 KB-aligned Slab, Scratch, Buddy, and Large sub-arenas without
    /// dereferencing device memory on the CPU.
    /// </summary>
    /// <param name="baseAddress">Base device or pinned pointer of the backing arena.</param>
    /// <param name="byteLength">Total capacity of the external arena in bytes (minimum 1 MB).</param>
    /// <param name="releaseAction">Optional callback invoked once with <paramref name="baseAddress"/> upon <see cref="Dispose"/>.</param>
    public DeviceMemoryPool(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        if (baseAddress == IntPtr.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(baseAddress), "External arena base address cannot be zero.");
        }

        if (byteLength < MinimumExternalArenaBytes || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(byteLength),
                $"External pool arena must be at least {MinimumExternalArenaBytes} bytes (1 MB) and within addressable bounds.");
        }

        _externalBase = baseAddress;
        _releaseAction = releaseAction;

        // Partition into 64 KB-aligned non-overlapping sub-arenas:
        // ~12.5% Slab (min 192 KB, max 64 MB), ~12.5% Scratch (min 128 KB, max 64 MB), ~50% Buddy, ~25% Large tail
        ulong slabBytes = AlignUp64K(Math.Clamp(byteLength / 8UL, 196_608UL, 64UL * 1024 * 1024));
        ulong scratchBytes = AlignUp64K(Math.Clamp(byteLength / 8UL, 131_072UL, 64UL * 1024 * 1024));
        ulong remaining = byteLength - slabBytes - scratchBytes;

        ulong buddyBytes = ((remaining / 2UL) / BuddyAllocator.MinBlockSize) * BuddyAllocator.MinBlockSize;
        if (buddyBytes < BuddyAllocator.MinBlockSize)
        {
            buddyBytes = BuddyAllocator.MinBlockSize;
        }

        ulong reservoirBytes = remaining - buddyBytes;

        IntPtr slabBase = baseAddress;
        IntPtr scratchBase = slabBase + (nint)slabBytes;
        IntPtr buddyBase = scratchBase + (nint)scratchBytes;
        IntPtr reservoirBase = buddyBase + (nint)buddyBytes;

        _slabAllocator = new SlabAllocator(slabBase, slabBytes);
        _scratchWorkspace = new RingScratchWorkspace(scratchBase, scratchBytes);
        _buddyAllocator = new BuddyAllocator(buddyBase, buddyBytes);
        _virtualReservoir = new VirtualMemoryReservoir(reservoirBase, reservoirBytes);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceMemoryPool"/> class from explicitly configured tier sub-allocators.
    /// </summary>
    public DeviceMemoryPool(
        SlabAllocator slabAllocator,
        BuddyAllocator buddyAllocator,
        VirtualMemoryReservoir virtualReservoir,
        RingScratchWorkspace scratchWorkspace)
    {
        ArgumentNullException.ThrowIfNull(slabAllocator);
        ArgumentNullException.ThrowIfNull(buddyAllocator);
        ArgumentNullException.ThrowIfNull(virtualReservoir);
        ArgumentNullException.ThrowIfNull(scratchWorkspace);

        _slabAllocator = slabAllocator;
        _buddyAllocator = buddyAllocator;
        _virtualReservoir = virtualReservoir;
        _scratchWorkspace = scratchWorkspace;
    }

    private static ulong AlignUp64K(ulong value) => (value + 65535UL) & ~65535UL;

    /// <inheritdoc/>
    public DevicePointer Allocate(ulong byteCount, MemoryCategory category)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (byteCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Allocation size must be greater than zero.");
        }

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
                if (byteCount <= BuddyAllocator.MaxBlockSize && byteCount <= _buddyAllocator.TotalCapacity)
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
        if (Volatile.Read(ref _disposed) != 0 || ptr.IsNull)
        {
            return;
        }

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

        if (_scratchWorkspace.Contains(ptr.Address))
        {
            _scratchWorkspace.Free(ptr);
            return;
        }
    }

    /// <inheritdoc/>
    public ScratchWorkspace RentScratchWorkspace(ulong minimumBytes)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var ptr = _scratchWorkspace.Allocate(minimumBytes);
        return new ScratchWorkspace(ptr.Address, ptr.SizeInBytes, this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _slabAllocator.Dispose();
            _buddyAllocator.Dispose();
            _virtualReservoir.Dispose();
            _scratchWorkspace.Dispose();

            IntPtr extBase = Interlocked.Exchange(ref _externalBase, IntPtr.Zero);
            if (extBase != IntPtr.Zero)
            {
                _releaseAction?.Invoke(extBase);
            }
        }
    }
}

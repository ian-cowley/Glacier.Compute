// <copyright file="SlabAllocator.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Tier 1: Fixed-Size Small Slabs (256 B – 64 KB).
/// Employs lock-free 64-bit atomic bitmask sub-allocation across either caller-supplied GPU/pinned arenas
/// (e.g. <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>D3D12</c>) or lazily materialized unmanaged host slabs.
/// </summary>
public sealed class SlabAllocator : IDisposable
{
    /// <summary>Pre-defined slab bin sizes (in bytes).</summary>
    public static readonly int[] BinSizes = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };

    /// <summary>Sum of one slot across all nine slab bin sizes (130,816 bytes).</summary>
    public const ulong SingleSlotAcrossAllBinsBytes = 130_816UL;

    private readonly SlabBin[] _bins;
    private IntPtr _externalBase;
    private readonly Action<IntPtr>? _releaseAction;
    private int _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="SlabAllocator"/> class backed by lazily allocated unmanaged host slabs.
    /// Individual bin arenas are allocated on first demand without upfront physical page zeroing.
    /// </summary>
    /// <param name="slotsPerBin">Number of fixed-size slots per bin (must be a positive multiple of 64; defaults to 1024).</param>
    public SlabAllocator(int slotsPerBin = 1024)
    {
        if (slotsPerBin > (int.MaxValue / 65536))
        {
            throw new ArgumentOutOfRangeException(nameof(slotsPerBin), "Slots per bin exceeds allowable memory capacity.");
        }

        if (slotsPerBin <= 0 || (slotsPerBin % 64) != 0)
        {
            slotsPerBin = 1024;
        }

        _bins = new SlabBin[BinSizes.Length];
        for (int i = 0; i < BinSizes.Length; i++)
        {
            _bins[i] = new SlabBin(BinSizes[i], slotsPerBin);
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SlabAllocator"/> class backed by an external device or pinned memory arena.
    /// Partitions <paramref name="baseAddress"/> across all nine slab bin sizes without dereferencing device memory on the CPU.
    /// </summary>
    /// <param name="baseAddress">Base device or pinned pointer of the backing arena.</param>
    /// <param name="byteLength">Total capacity of the external arena in bytes (at least 130,816 bytes).</param>
    /// <param name="releaseAction">Optional callback invoked with <paramref name="baseAddress"/> upon <see cref="Dispose"/>.</param>
    public SlabAllocator(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        if (baseAddress == IntPtr.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(baseAddress), "External arena base address cannot be zero.");
        }

        if (byteLength < SingleSlotAcrossAllBinsBytes || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(byteLength),
                $"External slab arena must be at least {SingleSlotAcrossAllBinsBytes} bytes and within addressable bounds.");
        }

        int slotsPerBin = (int)Math.Min(byteLength / SingleSlotAcrossAllBinsBytes, (ulong)(int.MaxValue / 65536));
        _externalBase = baseAddress;
        _releaseAction = releaseAction;
        _bins = new SlabBin[BinSizes.Length];

        ulong runningOffset = 0;
        for (int i = 0; i < BinSizes.Length; i++)
        {
            IntPtr binBase = baseAddress + (nint)runningOffset;
            _bins[i] = new SlabBin(BinSizes[i], slotsPerBin, binBase);
            runningOffset += (ulong)BinSizes[i] * (ulong)slotsPerBin;
        }
    }

    /// <summary>
    /// Sub-allocates a micro-slab buffer of the requested byte size (1 – 65,536 bytes).
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (byteCount == 0 || byteCount > 65536)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "SlabAllocator handles requests from 1 to 65,536 bytes.");
        }

        for (int i = 0; i < _bins.Length; i++)
        {
            if ((ulong)_bins[i].SlotSize >= byteCount)
            {
                if (_bins[i].TryAllocate(out IntPtr ptr))
                {
                    return new DevicePointer(ptr, (ulong)_bins[i].SlotSize);
                }
            }
        }

        throw new OutOfMemoryException(
            $"SlabAllocator exhausted for requested size {byteCount} bytes. Active allocated: {ActiveAllocatedBytes} bytes, Total committed: {TotalCommittedBytes} bytes.");
    }

    /// <summary>
    /// Frees a previously allocated slab pointer.
    /// </summary>
    public bool Free(DevicePointer ptr)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (ptr.IsNull)
        {
            return false;
        }

        for (int i = 0; i < _bins.Length; i++)
        {
            if (_bins[i].Contains(ptr.Address))
            {
                return _bins[i].Free(ptr.Address);
            }
        }

        return false;
    }

    /// <summary>
    /// Determines whether the given pointer belongs to any materialized slab bin in this allocator.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (address == IntPtr.Zero)
        {
            return false;
        }

        for (int i = 0; i < _bins.Length; i++)
        {
            if (_bins[i].Contains(address))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Gets total memory committed across all materialized slab bins.</summary>
    public ulong TotalCommittedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ulong total = 0;
            for (int i = 0; i < _bins.Length; i++)
            {
                total += _bins[i].CommittedBytes;
            }

            return total;
        }
    }

    /// <summary>Gets active allocated memory across all slab bins.</summary>
    public ulong ActiveAllocatedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ulong total = 0;
            for (int i = 0; i < _bins.Length; i++)
            {
                total += _bins[i].AllocatedBytes;
            }

            return total;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            for (int i = 0; i < _bins.Length; i++)
            {
                _bins[i].Dispose();
            }

            IntPtr extBase = Interlocked.Exchange(ref _externalBase, IntPtr.Zero);
            if (extBase != IntPtr.Zero)
            {
                _releaseAction?.Invoke(extBase);
            }
        }
    }

    /// <summary>
    /// Individual slab bin managing fixed-size slots via lock-free 64-bit atomic masks.
    /// </summary>
    private sealed unsafe class SlabBin : IDisposable
    {
        public readonly int SlotSize;
        public readonly int SlotCount;
        public readonly ulong TotalBytes;

        private readonly object _initLock = new();
        private readonly bool _ownsHostMemory;
        private readonly long[] _bitmasks; // 0 = free, 1 = occupied
        private IntPtr _baseAddress;
        private int _allocatedCount;
        private bool _disposed;

        public ulong AllocatedBytes => (ulong)Volatile.Read(ref _allocatedCount) * (ulong)SlotSize;

        public ulong CommittedBytes => Volatile.Read(ref _baseAddress) != IntPtr.Zero ? TotalBytes : 0UL;

        /// <summary>Lazy host unmanaged slab constructor.</summary>
        public SlabBin(int slotSize, int slotCount)
        {
            SlotSize = slotSize;
            SlotCount = slotCount;
            TotalBytes = (ulong)slotSize * (ulong)slotCount;
            _ownsHostMemory = true;
            _bitmasks = CreateBitmasks(slotCount);
        }

        /// <summary>External pre-partitioned device/pinned slab constructor.</summary>
        public SlabBin(int slotSize, int slotCount, IntPtr externalBaseAddress)
        {
            SlotSize = slotSize;
            SlotCount = slotCount;
            TotalBytes = (ulong)slotSize * (ulong)slotCount;
            _ownsHostMemory = false;
            _baseAddress = externalBaseAddress;
            _bitmasks = CreateBitmasks(slotCount);
        }

        private static long[] CreateBitmasks(int slotCount)
        {
            int wordCount = (slotCount + 63) / 64;
            long[] masks = new long[wordCount];
            int rem = slotCount % 64;
            if (rem != 0)
            {
                // Pre-mark trailing bits beyond slotCount as occupied (1)
                masks[wordCount - 1] = unchecked((long)~((1UL << rem) - 1UL));
            }

            return masks;
        }

        private IntPtr EnsureMaterialized()
        {
            IntPtr existing = Volatile.Read(ref _baseAddress);
            if (existing != IntPtr.Zero || !_ownsHostMemory)
            {
                return existing;
            }

            lock (_initLock)
            {
                if (_disposed)
                {
                    return IntPtr.Zero;
                }

                if (_baseAddress == IntPtr.Zero)
                {
                    IntPtr allocated = (IntPtr)NativeMemory.AlignedAlloc((nuint)TotalBytes, 64);
                    Volatile.Write(ref _baseAddress, allocated);
                }

                return _baseAddress;
            }
        }

        public bool Contains(IntPtr address)
        {
            IntPtr baseAddr = Volatile.Read(ref _baseAddress);
            if (_disposed || address == IntPtr.Zero || baseAddr == IntPtr.Zero)
            {
                return false;
            }

            nint diff = address - baseAddr;
            return diff >= 0 && diff < (nint)TotalBytes;
        }

        public bool TryAllocate(out IntPtr ptr)
        {
            if (_disposed)
            {
                ptr = IntPtr.Zero;
                return false;
            }

            IntPtr baseAddr = EnsureMaterialized();
            if (baseAddr == IntPtr.Zero)
            {
                ptr = IntPtr.Zero;
                return false;
            }

            for (int wordIdx = 0; wordIdx < _bitmasks.Length; wordIdx++)
            {
                while (true)
                {
                    long current = Volatile.Read(ref _bitmasks[wordIdx]);
                    if (current == -1L)
                    {
                        break;
                    }

                    ulong inverted = ~(ulong)current;
                    int bitIndex = BitOperations.TrailingZeroCount(inverted);
                    long mask = 1L << bitIndex;

                    long updated = current | mask;
                    if (Interlocked.CompareExchange(ref _bitmasks[wordIdx], updated, current) == current)
                    {
                        Interlocked.Increment(ref _allocatedCount);
                        long globalSlot = ((long)wordIdx * 64L) + bitIndex;
                        ptr = baseAddr + (nint)(globalSlot * SlotSize);
                        return true;
                    }
                }
            }

            ptr = IntPtr.Zero;
            return false;
        }

        public bool Free(IntPtr address)
        {
            IntPtr baseAddr = Volatile.Read(ref _baseAddress);
            if (_disposed || baseAddr == IntPtr.Zero)
            {
                return false;
            }

            nint diff = address - baseAddr;
            if (diff < 0 || diff >= (nint)TotalBytes || (diff % SlotSize) != 0)
            {
                return false;
            }

            int globalSlot = (int)(diff / SlotSize);
            if (globalSlot >= SlotCount)
            {
                return false;
            }

            int wordIdx = globalSlot / 64;
            int bitIndex = globalSlot % 64;
            long mask = 1L << bitIndex;

            while (true)
            {
                long current = Volatile.Read(ref _bitmasks[wordIdx]);
                if ((current & mask) == 0)
                {
                    return false;
                }

                long updated = current & ~mask;
                if (Interlocked.CompareExchange(ref _bitmasks[wordIdx], updated, current) == current)
                {
                    Interlocked.Decrement(ref _allocatedCount);
                    return true;
                }
            }
        }

        public void Dispose()
        {
            lock (_initLock)
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;
                IntPtr baseAddr = Interlocked.Exchange(ref _baseAddress, IntPtr.Zero);
                if (_ownsHostMemory && baseAddr != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree((void*)baseAddr);
                }
            }
        }
    }
}

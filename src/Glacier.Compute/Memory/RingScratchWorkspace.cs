// <copyright file="RingScratchWorkspace.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Circular scratch ring sub-allocator for high-frequency stream workspaces with active-lease wraparound protection.
/// Supports sub-allocating caller-supplied device/pinned arenas (e.g. <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>D3D12</c>)
/// or standalone aligned unmanaged host arenas using exact physical interval accounting and Vyukov per-slot turn tracking.
/// </summary>
public sealed class RingScratchWorkspace : IDisposable
{
    private const ulong MaxCapacityBytes = 1UL << 46;

    [StructLayout(LayoutKind.Explicit, Size = 128)]
    private struct PaddedCursor
    {
        [FieldOffset(64)]
        public ulong Value;
    }

    [StructLayout(LayoutKind.Explicit, Size = 64)]
    private struct LeaseSlot
    {
        [FieldOffset(0)]
        public long Turn;

        [FieldOffset(8)]
        public int State; // 0 = Empty, 1 = Active, 2 = Released

        [FieldOffset(16)]
        public IntPtr Address;

        [FieldOffset(24)]
        public ulong StartInRing;

        [FieldOffset(32)]
        public ulong Size;
    }

    private IntPtr _baseAddress;
    private readonly ulong _capacity;
    private readonly int _leaseCapacity;
    private readonly int _leaseMask;
    private readonly LeaseSlot[] _leases;
    private readonly bool _ownsHostMemory;
    private readonly Action<IntPtr>? _releaseAction;
    private readonly object _sync = new();
    private PaddedCursor _readOffset;
    private ulong _writePos;
    private ulong _headSeq;
    private ulong _tailSeq;
    private int _activeCount;
    private int _disposed;

    /// <summary>Gets the base native or device address of the backing ring arena.</summary>
    public IntPtr BaseAddress
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return Volatile.Read(ref _baseAddress);
        }
    }

    /// <summary>Gets the total buffer capacity in bytes.</summary>
    public ulong Capacity
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _capacity;
        }
    }

    /// <summary>Gets the current byte offset inside the circular buffer.</summary>
    public ulong CurrentOffset
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return Volatile.Read(ref _writePos);
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RingScratchWorkspace"/> class backed by an aligned unmanaged host memory arena.
    /// Does not eagerly fault physical pages across the entire buffer at construction time.
    /// </summary>
    /// <param name="capacityBytes">Total host ring capacity in bytes (minimum 64 bytes, default 64 MB).</param>
    public unsafe RingScratchWorkspace(ulong capacityBytes = 64UL * 1024 * 1024)
    {
        if (capacityBytes < 64 || capacityBytes > MaxCapacityBytes || capacityBytes > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityBytes), "Ring capacity must be between 64 bytes and 64 TB within addressable memory bounds.");
        }

        _capacity = capacityBytes;
        _leaseCapacity = ComputeLeaseCapacity(_capacity);
        _leaseMask = _leaseCapacity - 1;
        _leases = InitializeLeaseSlots(_leaseCapacity);
        _baseAddress = (IntPtr)NativeMemory.AlignedAlloc((nuint)_capacity, 64);
        _ownsHostMemory = true;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RingScratchWorkspace"/> class backed by an external device or pinned arena
    /// (such as CUDA <c>cuMemAlloc</c> / <c>cuMemHostAlloc</c> or Direct3D 12 / Vulkan memory).
    /// Never dereferences <paramref name="baseAddress"/> on the CPU.
    /// </summary>
    /// <param name="baseAddress">Base device or pinned pointer of the backing arena.</param>
    /// <param name="byteLength">Total capacity of the external arena in bytes (minimum 64 bytes).</param>
    /// <param name="releaseAction">Optional callback invoked with <paramref name="baseAddress"/> upon <see cref="Dispose"/>.</param>
    public RingScratchWorkspace(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        if (baseAddress == IntPtr.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(baseAddress), "External arena base address cannot be zero.");
        }

        if (byteLength < 64 || byteLength > MaxCapacityBytes || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength), "External ring capacity must be between 64 bytes and 64 TB within addressable bounds.");
        }

        _baseAddress = baseAddress;
        _capacity = byteLength;
        _leaseCapacity = ComputeLeaseCapacity(_capacity);
        _leaseMask = _leaseCapacity - 1;
        _leases = InitializeLeaseSlots(_leaseCapacity);
        _ownsHostMemory = false;
        _releaseAction = releaseAction;
    }

    private static int ComputeLeaseCapacity(ulong capacity)
    {
        uint minSlots = (uint)Math.Clamp(capacity / 64UL, 64UL, 4096UL);
        return (int)BitOperations.RoundUpToPowerOf2(minSlots);
    }

    private static LeaseSlot[] InitializeLeaseSlots(int leaseCapacity)
    {
        var slots = new LeaseSlot[leaseCapacity];
        for (int i = 0; i < leaseCapacity; i++)
        {
            slots[i].Turn = i;
        }

        return slots;
    }

    /// <summary>
    /// Allocates a contiguous 64-byte-aligned scratch slice from the ring buffer.
    /// Throws <see cref="OutOfMemoryException"/> if active unreleased leases prevent safe wraparound.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        if (TryAllocate(byteCount, out DevicePointer ptr))
        {
            return ptr;
        }

        throw new OutOfMemoryException(
            $"RingScratchWorkspace saturated (capacity {_capacity} bytes): active leases prevent wraparound without overwriting live slices.");
    }

    /// <summary>
    /// Allocates a contiguous 64-byte-aligned scratch slice, waiting up to <paramref name="timeout"/> for active leases to release.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount, TimeSpan timeout, CancellationToken ct = default)
    {
        if (TryAllocate(byteCount, timeout, out DevicePointer ptr, ct))
        {
            return ptr;
        }

        throw new TimeoutException($"Timed out after {timeout} waiting for RingScratchWorkspace lease capacity.");
    }

    /// <summary>
    /// Attempts to allocate a contiguous 64-byte-aligned scratch slice without blocking.
    /// </summary>
    public bool TryAllocate(ulong byteCount, out DevicePointer pointer)
    {
        return TryAllocate(byteCount, TimeSpan.Zero, out pointer, CancellationToken.None);
    }

    /// <summary>
    /// Attempts to allocate a contiguous 64-byte-aligned scratch slice, waiting up to <paramref name="timeout"/> if saturated.
    /// </summary>
    public bool TryAllocate(ulong byteCount, TimeSpan timeout, out DevicePointer pointer, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();

        if (byteCount == 0 || byteCount > ulong.MaxValue - 63UL)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), $"Requested size {byteCount} is invalid.");
        }

        if (timeout != Timeout.InfiniteTimeSpan && timeout < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be non-negative or Timeout.InfiniteTimeSpan.");
        }

        ulong alignedSize = (byteCount + 63UL) & ~63UL;
        if (alignedSize > _capacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(byteCount),
                $"Aligned allocation size {alignedSize} bytes (requested {byteCount} bytes) exceeds ring capacity {_capacity} bytes.");
        }

        bool infinite = timeout == Timeout.InfiniteTimeSpan;
        bool hasTimeout = infinite || timeout > TimeSpan.Zero;
        long startTicks = (!infinite && hasTimeout) ? Stopwatch.GetTimestamp() : 0L;

        using CancellationTokenRegistration ctr = (hasTimeout && ct.CanBeCanceled)
            ? ct.UnsafeRegister(
                static state =>
                {
                    object syncObj = state!;
                    lock (syncObj)
                    {
                        Monitor.PulseAll(syncObj);
                    }
                },
                _sync)
            : default;

        lock (_sync)
        {
            while (true)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                ct.ThrowIfCancellationRequested();

                IntPtr baseAddr = Volatile.Read(ref _baseAddress);
                ObjectDisposedException.ThrowIf(baseAddr == IntPtr.Zero, this);

                int slotIdx = (int)(_headSeq & (ulong)_leaseMask);
                bool canReserveSlot = _activeCount < _leaseCapacity
                    && (_headSeq - _tailSeq) < (ulong)_leaseCapacity
                    && Volatile.Read(ref _leases[slotIdx].Turn) == (long)_headSeq;

                bool hasSpace = false;
                ulong startInRing = 0;
                ulong newWritePos = 0;

                if (canReserveSlot)
                {
                    if (_activeCount == 0)
                    {
                        if (_writePos + alignedSize <= _capacity)
                        {
                            startInRing = _writePos;
                            newWritePos = (_writePos + alignedSize) % _capacity;
                            hasSpace = true;
                        }
                        else
                        {
                            // Ring is empty and alignedSize <= _capacity: wrap cleanly to offset 0.
                            startInRing = 0;
                            newWritePos = alignedSize % _capacity;
                            hasSpace = true;
                        }
                    }
                    else
                    {
                        ulong tailStart = _leases[(int)(_tailSeq & (ulong)_leaseMask)].StartInRing;
                        if (tailStart == _writePos)
                        {
                            // Active leases occupy the entire ring.
                            hasSpace = false;
                        }
                        else if (tailStart < _writePos)
                        {
                            // Active region is [tailStart, _writePos).
                            if (_writePos + alignedSize <= _capacity)
                            {
                                startInRing = _writePos;
                                newWritePos = (_writePos + alignedSize) % _capacity;
                                hasSpace = true;
                            }
                            else if (alignedSize <= tailStart)
                            {
                                // Wrap cleanly into [0, alignedSize) without overlapping [tailStart, _writePos).
                                startInRing = 0;
                                newWritePos = alignedSize;
                                hasSpace = true;
                            }
                        }
                        else
                        {
                            // Active region is wrapped; free contiguous space is [_writePos, tailStart).
                            if (_writePos + alignedSize <= tailStart)
                            {
                                startInRing = _writePos;
                                newWritePos = _writePos + alignedSize;
                                hasSpace = true;
                            }
                        }
                    }
                }

                if (hasSpace)
                {
                    IntPtr ptr = baseAddr + (nint)startInRing;
                    _leases[slotIdx].Address = ptr;
                    _leases[slotIdx].StartInRing = startInRing;
                    _leases[slotIdx].Size = alignedSize;
                    Volatile.Write(ref _leases[slotIdx].State, 1);

                    _headSeq++;
                    _activeCount++;
                    Volatile.Write(ref _writePos, newWritePos);
                    if (_activeCount == 1)
                    {
                        Volatile.Write(ref _readOffset.Value, startInRing);
                    }

                    pointer = new DevicePointer(ptr, alignedSize);
                    return true;
                }

                if (!hasTimeout)
                {
                    pointer = DevicePointer.Null;
                    return false;
                }

                if (infinite)
                {
                    Monitor.Wait(_sync);
                }
                else
                {
                    TimeSpan elapsed = Stopwatch.GetElapsedTime(startTicks);
                    if (elapsed >= timeout)
                    {
                        pointer = DevicePointer.Null;
                        return false;
                    }

                    TimeSpan remaining = timeout - elapsed;
                    TimeSpan waitSlice = remaining.TotalMilliseconds > int.MaxValue
                        ? TimeSpan.FromMinutes(1)
                        : remaining;
                    Monitor.Wait(_sync, waitSlice);
                }
            }
        }
    }

    /// <summary>
    /// Releases an active scratch slice lease, allowing ring capacity to advance and wrap safely.
    /// Supports out-of-order release across concurrent threads.
    /// </summary>
    public bool Free(DevicePointer ptr)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (ptr.IsNull || !Contains(ptr.Address))
        {
            return false;
        }

        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            bool found = false;
            for (ulong s = _tailSeq; s < _headSeq; s++)
            {
                int idx = (int)(s & (ulong)_leaseMask);
                if (Volatile.Read(ref _leases[idx].State) == 1 && _leases[idx].Address == ptr.Address)
                {
                    Volatile.Write(ref _leases[idx].State, 2);
                    _activeCount--;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return false;
            }

            while (_tailSeq < _headSeq)
            {
                int tailIdx = (int)(_tailSeq & (ulong)_leaseMask);
                if (Volatile.Read(ref _leases[tailIdx].State) != 2)
                {
                    break;
                }

                _leases[tailIdx].Address = IntPtr.Zero;
                _leases[tailIdx].StartInRing = 0;
                _leases[tailIdx].Size = 0;
                Volatile.Write(ref _leases[tailIdx].State, 0);
                Volatile.Write(ref _leases[tailIdx].Turn, (long)(_tailSeq + (ulong)_leaseCapacity));
                _tailSeq++;
            }

            ulong newReadOffset = _activeCount == 0
                ? _writePos
                : _leases[(int)(_tailSeq & (ulong)_leaseMask)].StartInRing;
            Volatile.Write(ref _readOffset.Value, newReadOffset);
            Monitor.PulseAll(_sync);
            return true;
        }
    }

    /// <summary>
    /// Checks if a pointer resides within this ring scratch workspace.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        IntPtr baseAddr = Volatile.Read(ref _baseAddress);
        if (address == IntPtr.Zero || baseAddr == IntPtr.Zero)
        {
            return false;
        }

        nint diff = address - baseAddr;
        return diff >= 0 && diff < (nint)_capacity;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            IntPtr baseAddr = Interlocked.Exchange(ref _baseAddress, IntPtr.Zero);
            lock (_sync)
            {
                Monitor.PulseAll(_sync);
            }

            if (baseAddr != IntPtr.Zero)
            {
                if (_ownsHostMemory)
                {
                    NativeMemory.AlignedFree((void*)baseAddr);
                }
                else
                {
                    _releaseAction?.Invoke(baseAddr);
                }
            }
        }
    }
}

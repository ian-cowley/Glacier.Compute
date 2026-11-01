// <copyright file="RingScratchWorkspace.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Circular lock-free scratch ring sub-allocator for high-frequency stream workspaces.
/// Supports sub-allocating caller-supplied device/pinned arenas (e.g. <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>D3D12</c>)
/// or standalone aligned unmanaged host arenas in ~15 ns using a single atomic compare-and-swap pointer bump.
/// </summary>
public sealed class RingScratchWorkspace : IDisposable
{
    private IntPtr _baseAddress;
    private readonly ulong _capacity;
    private readonly bool _ownsHostMemory;
    private readonly Action<IntPtr>? _releaseAction;
    private long _writeOffset;
    private int _disposed;

    /// <summary>Gets the base native or device address of the backing ring arena.</summary>
    public IntPtr BaseAddress
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _baseAddress;
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
            return (ulong)Volatile.Read(ref _writeOffset) % _capacity;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RingScratchWorkspace"/> class backed by an aligned unmanaged host memory arena.
    /// Does not eagerly fault physical pages across the entire buffer at construction time.
    /// </summary>
    /// <param name="capacityBytes">Total host ring capacity in bytes (minimum 64 bytes, default 64 MB).</param>
    public unsafe RingScratchWorkspace(ulong capacityBytes = 64UL * 1024 * 1024)
    {
        if (capacityBytes < 64 || capacityBytes > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityBytes), "Ring capacity must be at least 64 bytes and within addressable memory bounds.");
        }

        _capacity = capacityBytes;
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

        if (byteLength < 64 || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength), "External ring capacity must be at least 64 bytes and within addressable bounds.");
        }

        _baseAddress = baseAddress;
        _capacity = byteLength;
        _ownsHostMemory = false;
        _releaseAction = releaseAction;
    }

    /// <summary>
    /// Allocates a contiguous 64-byte-aligned scratch slice from the ring buffer using a lock-free atomic compare-and-swap loop.
    /// Guarantees non-overlapping slices across concurrent callers until total ring capacity wraps.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (byteCount == 0 || byteCount > ulong.MaxValue - 63UL)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), $"Requested size {byteCount} is invalid.");
        }

        ulong alignedSize = (byteCount + 63UL) & ~63UL;
        if (alignedSize > _capacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(byteCount),
                $"Aligned allocation size {alignedSize} bytes (requested {byteCount} bytes) exceeds ring capacity {_capacity} bytes.");
        }

        while (true)
        {
            long current = Volatile.Read(ref _writeOffset);
            ulong pos = (ulong)current % _capacity;
            ulong pad = (pos + alignedSize > _capacity) ? (_capacity - pos) : 0UL;
            ulong startInRing = (pad == 0UL) ? pos : 0UL;
            long next = current + (long)(pad + alignedSize);

            if (Interlocked.CompareExchange(ref _writeOffset, next, current) == current)
            {
                IntPtr ptr = _baseAddress + (nint)startInRing;
                return new DevicePointer(ptr, alignedSize);
            }
        }
    }

    /// <summary>
    /// Checks if a pointer resides within this ring scratch workspace.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (address == IntPtr.Zero || _baseAddress == IntPtr.Zero)
        {
            return false;
        }

        nint diff = address - _baseAddress;
        return diff >= 0 && diff < (nint)_capacity;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            IntPtr baseAddr = Interlocked.Exchange(ref _baseAddress, IntPtr.Zero);
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

// <copyright file="RingScratchWorkspace.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Dedicated circular lock-free scratch ring buffer for high-frequency stream workspaces.
/// Provides nanosecond-level pointer-bump allocations for temporary activations and gradients.
/// </summary>
public sealed class RingScratchWorkspace : IDisposable
{
    private readonly IntPtr _baseAddress;
    private readonly ulong _capacity;
    private long _writeOffset;
    private bool _disposed;

    /// <summary>Gets the total buffer capacity in bytes.</summary>
    public ulong Capacity => _capacity;

    /// <summary>Gets the current byte offset inside the circular buffer.</summary>
    public ulong CurrentOffset => (ulong)(Volatile.Read(ref _writeOffset) % (long)_capacity);

    /// <summary>
    /// Initializes a new instance of the <see cref="RingScratchWorkspace"/> class.
    /// </summary>
    /// <param name="capacityBytes">Total ring capacity (default 64 MB).</param>
    public unsafe RingScratchWorkspace(ulong capacityBytes = 64UL * 1024 * 1024)
    {
        _capacity = capacityBytes;
        _baseAddress = (IntPtr)NativeMemory.AlignedAlloc((nuint)_capacity, 64);
        NativeMemory.Clear((void*)_baseAddress, (nuint)_capacity);
    }

    /// <summary>
    /// Allocates a contiguous scratch slice from the ring buffer using a lock-free atomic pointer bump.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (byteCount == 0 || byteCount > _capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), $"Requested size {byteCount} exceeds ring capacity {_capacity}.");
        }

        // Align allocation to 64 bytes
        ulong alignedSize = (byteCount + 63) & ~63UL;

        // Atomic pointer bump
        long oldOffset = Interlocked.Add(ref _writeOffset, (long)alignedSize) - (long)alignedSize;
        ulong startInRing = (ulong)oldOffset % _capacity;

        // Wrap check: if allocation wraps around the boundary, bump again to start from 0
        if (startInRing + alignedSize > _capacity)
        {
            // Reset to beginning of ring
            oldOffset = Interlocked.Add(ref _writeOffset, (long)alignedSize) - (long)alignedSize;
            startInRing = 0;
        }

        IntPtr ptr = _baseAddress + (nint)startInRing;
        return new DevicePointer(ptr, alignedSize);
    }

    /// <summary>
    /// Checks if a pointer resides within this ring scratch workspace.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        nint diff = address - _baseAddress;
        return diff >= 0 && diff < (nint)_capacity;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_baseAddress != IntPtr.Zero)
            {
                NativeMemory.AlignedFree((void*)_baseAddress);
            }
        }
    }
}

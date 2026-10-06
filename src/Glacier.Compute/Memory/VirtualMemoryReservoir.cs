// <copyright file="VirtualMemoryReservoir.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

/// <summary>
/// Tier 3: Virtual Memory Page Reservoir (> 32 MB).
/// Reserves large virtual address spaces and commits physical pages on-demand for large model weights.
/// Eliminates address-space fragmentation and provides zero-leak deterministic cleanup.
/// </summary>
public sealed class VirtualMemoryReservoir : IDisposable
{
    private const uint MEM_COMMIT = 0x00001000;
    private const uint MEM_RESERVE = 0x00002000;
    private const uint MEM_DECOMMIT = 0x00004000;
    private const uint MEM_RELEASE = 0x00008000;
    private const uint PAGE_READWRITE = 0x04;
    private const uint PAGE_NOACCESS = 0x01;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool VirtualFree(IntPtr lpAddress, nuint dwSize, uint dwFreeType);

    private readonly object _lock = new();
    private IntPtr _reservedBase;
    private readonly ulong _totalReservedBytes;
    private ulong _committedOffset;
    private readonly Dictionary<IntPtr, ulong> _committedAllocations = new();
    private bool _disposed;
    private bool _isWindows;

    /// <summary>Gets the total reserved virtual address space capacity in bytes.</summary>
    public ulong TotalReservedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _totalReservedBytes;
        }
    }

    /// <summary>Gets the currently committed memory in bytes.</summary>
    public ulong TotalCommittedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            lock (_lock)
            {
                return _committedOffset;
            }
        }
    }

    /// <summary>Gets the currently active allocated memory in bytes.</summary>
    public ulong ActiveAllocatedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            lock (_lock)
            {
                ulong sum = 0;
                foreach (var val in _committedAllocations.Values)
                {
                    sum += val;
                }

                return sum;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VirtualMemoryReservoir"/> class.
    /// </summary>
    /// <param name="reserveBytes">Total virtual address range to reserve (e.g. 16 GB, default 4 GB).</param>
    public unsafe VirtualMemoryReservoir(ulong reserveBytes = 4UL * 1024 * 1024 * 1024)
    {
        if (reserveBytes == 0 || reserveBytes > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(reserveBytes), "Reserved capacity must be greater than zero and within addressable memory range.");
        }

        _totalReservedBytes = reserveBytes;
        _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        if (_isWindows)
        {
            _reservedBase = VirtualAlloc(IntPtr.Zero, (nuint)reserveBytes, MEM_RESERVE, PAGE_NOACCESS);
            if (_reservedBase == IntPtr.Zero)
            {
                // Fallback to aligned unmanaged reservation
                _reservedBase = (IntPtr)NativeMemory.AlignedAlloc((nuint)reserveBytes, 65536);
                NativeMemory.Clear((void*)_reservedBase, (nuint)reserveBytes);
                _isWindows = false;
            }
        }
        else
        {
            _reservedBase = (IntPtr)NativeMemory.AlignedAlloc((nuint)reserveBytes, 65536);
            NativeMemory.Clear((void*)_reservedBase, (nuint)reserveBytes);
        }
    }

    /// <summary>
    /// Commits and maps physical memory for a requested block size from the reserved pool.
    /// Note: Memory allocation advances an append-only commit watermark across the reserved arena.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (byteCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Allocation size must be greater than zero.");
        }

        if (byteCount > _totalReservedBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), $"Requested size {byteCount} bytes exceeds total reserved capacity of {_totalReservedBytes} bytes.");
        }

        if (byteCount > ulong.MaxValue - 65535UL)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size causes integer overflow when aligning to page boundary.");
        }

        // Align up to 64 KB page boundary
        ulong alignedSize = (byteCount + 65535UL) & ~65535UL;

        lock (_lock)
        {
            if (_committedOffset + alignedSize > _totalReservedBytes)
            {
                throw new OutOfMemoryException($"Virtual memory reservoir exhausted: requested {alignedSize} bytes, committed {_committedOffset} bytes, reserved capacity {_totalReservedBytes} bytes.");
            }

            IntPtr address = _reservedBase + (nint)_committedOffset;
            if (_isWindows)
            {
                IntPtr result = VirtualAlloc(address, (nuint)alignedSize, MEM_COMMIT, PAGE_READWRITE);
                if (result == IntPtr.Zero)
                {
                    int win32Error = Marshal.GetLastWin32Error();
                    throw new OutOfMemoryException($"VirtualAlloc commit failed at address 0x{address:X16} for {alignedSize} bytes. Win32 error code: {win32Error} (0x{win32Error:X8}).");
                }
            }

            _committedOffset += alignedSize;
            _committedAllocations[address] = alignedSize;
            return new DevicePointer(address, alignedSize);
        }
    }

    /// <summary>
    /// Decommits memory associated with a previously allocated pointer.
    /// </summary>
    public bool Free(DevicePointer ptr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (ptr.IsNull)
        {
            return false;
        }

        if (!Contains(ptr.Address))
        {
            return false;
        }

        lock (_lock)
        {
            if (!_committedAllocations.Remove(ptr.Address, out ulong size))
            {
                return false;
            }

            if (_isWindows)
            {
                bool freed = VirtualFree(ptr.Address, (nuint)size, MEM_DECOMMIT);
                if (!freed)
                {
                    int win32Error = Marshal.GetLastWin32Error();
                    throw new InvalidOperationException($"VirtualFree decommit failed at address 0x{ptr.Address:X16} for {size} bytes. Win32 error code: {win32Error} (0x{win32Error:X8}).");
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Checks if a pointer lies within the reserved address range of this reservoir.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (address == IntPtr.Zero || _reservedBase == IntPtr.Zero)
        {
            return false;
        }

        nint diff = address - _reservedBase;
        return diff >= 0 && diff < (nint)_totalReservedBytes;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;
                _committedAllocations.Clear();
                IntPtr basePtr = _reservedBase;
                _reservedBase = IntPtr.Zero;
                if (basePtr != IntPtr.Zero)
                {
                    if (_isWindows)
                    {
                        VirtualFree(basePtr, 0, MEM_RELEASE);
                    }
                    else
                    {
                        NativeMemory.AlignedFree((void*)basePtr);
                    }
                }
            }
        }
    }
}

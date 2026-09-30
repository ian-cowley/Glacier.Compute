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
    private readonly IntPtr _reservedBase;
    private readonly ulong _totalReservedBytes;
    private ulong _committedOffset;
    private readonly Dictionary<IntPtr, ulong> _committedAllocations = new();
    private bool _disposed;
    private readonly bool _isWindows;

    /// <summary>Gets the total reserved virtual address space capacity in bytes.</summary>
    public ulong TotalReservedBytes => _totalReservedBytes;

    /// <summary>Gets the currently committed memory in bytes.</summary>
    public ulong TotalCommittedBytes
    {
        get
        {
            lock (_lock)
            {
                return _committedOffset;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VirtualMemoryReservoir"/> class.
    /// </summary>
    /// <param name="reserveBytes">Total virtual address range to reserve (e.g. 16 GB, default 4 GB).</param>
    public unsafe VirtualMemoryReservoir(ulong reserveBytes = 4UL * 1024 * 1024 * 1024)
    {
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
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (byteCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Allocation size must be greater than zero.");
        }

        // Align up to 64 KB page boundary
        ulong alignedSize = (byteCount + 65535) & ~65535UL;

        lock (_lock)
        {
            if (_committedOffset + alignedSize > _totalReservedBytes)
            {
                throw new OutOfMemoryException($"Virtual memory reservoir exhausted: requested {alignedSize} bytes, reserved capacity {_totalReservedBytes} bytes.");
            }

            IntPtr address = _reservedBase + (nint)_committedOffset;
            if (_isWindows)
            {
                IntPtr result = VirtualAlloc(address, (nuint)alignedSize, MEM_COMMIT, PAGE_READWRITE);
                if (result == IntPtr.Zero)
                {
                    throw new OutOfMemoryException($"VirtualAlloc commit failed at address 0x{address:X16} for {alignedSize} bytes.");
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
        if (_disposed || ptr.IsNull)
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
                VirtualFree(ptr.Address, (nuint)size, MEM_DECOMMIT);
            }

            return true;
        }
    }

    /// <summary>
    /// Checks if a pointer lies within the reserved address range of this reservoir.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        nint diff = address - _reservedBase;
        return diff >= 0 && diff < (nint)_totalReservedBytes;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            lock (_lock)
            {
                _committedAllocations.Clear();
                if (_reservedBase != IntPtr.Zero)
                {
                    if (_isWindows)
                    {
                        VirtualFree(_reservedBase, 0, MEM_RELEASE);
                    }
                    else
                    {
                        NativeMemory.AlignedFree((void*)_reservedBase);
                    }
                }
            }
        }
    }
}

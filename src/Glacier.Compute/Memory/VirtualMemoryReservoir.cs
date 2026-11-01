// <copyright file="VirtualMemoryReservoir.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

/// <summary>
/// Tier 3: Virtual Memory Page Reservoir (&gt; 32 MB) and Large Tail Arena Sub-Allocator.
/// Reserves zero-RSS virtual address space via <c>VirtualAlloc(MEM_RESERVE)</c> on Windows and
/// <c>mmap(PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS)</c> on Linux/macOS, committing physical pages strictly on-demand
/// and releasing them via <c>VirtualFree(MEM_DECOMMIT)</c> or <c>madvise(MADV_DONTNEED)</c> + <c>mprotect(PROT_NONE)</c>.
/// Also supports sub-allocating caller-supplied external device/pinned arenas without touching memory on the CPU.
/// </summary>
public sealed partial class VirtualMemoryReservoir : IDisposable
{
    private const uint MEM_COMMIT = 0x00001000;
    private const uint MEM_RESERVE = 0x00002000;
    private const uint MEM_DECOMMIT = 0x00004000;
    private const uint MEM_RELEASE = 0x00008000;
    private const uint PAGE_NOACCESS = 0x01;
    private const uint PAGE_READWRITE = 0x04;

    private const int PROT_NONE = 0x0;
    private const int PROT_READ = 0x1;
    private const int PROT_WRITE = 0x2;
    private const int MAP_PRIVATE = 0x02;
    private const int MAP_ANONYMOUS_LINUX = 0x20;
    private const int MAP_ANON_BSD = 0x1000;
    private const int MADV_DONTNEED = 4;

    private enum BackingMode : byte
    {
        WindowsVirtualAlloc = 0,
        UnixMmap = 1,
        HostAlignedFallback = 2,
        ExternalArena = 3
    }

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial IntPtr VirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool VirtualFree(IntPtr lpAddress, nuint dwSize, uint dwFreeType);

    [LibraryImport("libc", SetLastError = true)]
    private static partial IntPtr mmap(IntPtr addr, nuint length, int prot, int flags, int fd, long offset);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int mprotect(IntPtr addr, nuint len, int prot);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int madvise(IntPtr addr, nuint len, int advice);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int munmap(IntPtr addr, nuint length);

    private readonly object _lock = new();
    private IntPtr _reservedBase;
    private readonly ulong _totalReservedBytes;
    private readonly ulong _osMappingBytes;
    private readonly BackingMode _backingMode;
    private readonly Action<IntPtr>? _releaseAction;
    private ulong _committedOffset;
    private readonly Dictionary<IntPtr, ulong> _committedAllocations = new();
    private bool _disposed;

    /// <summary>Gets the base reserved or external address.</summary>
    public IntPtr BaseAddress
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _reservedBase;
        }
    }

    /// <summary>Gets the total reserved virtual address space capacity in bytes.</summary>
    public ulong TotalReservedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _totalReservedBytes;
        }
    }

    /// <summary>Gets the cumulative committed watermark in bytes.</summary>
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
    /// Initializes a new instance of the <see cref="VirtualMemoryReservoir"/> class with zero-RSS virtual address space reservation.
    /// </summary>
    /// <param name="reserveBytes">Total host virtual address range to reserve (default 4 GB; zero physical RSS until <see cref="Allocate"/>).</param>
    public unsafe VirtualMemoryReservoir(ulong reserveBytes = 4UL * 1024 * 1024 * 1024)
    {
        ulong granularity = Math.Max((ulong)Environment.SystemPageSize, 65536UL);
        if (reserveBytes == 0 || reserveBytes > (ulong)nint.MaxValue - (granularity - 1UL))
        {
            throw new ArgumentOutOfRangeException(nameof(reserveBytes), "Reserved capacity must be greater than zero and within addressable memory range.");
        }

        _totalReservedBytes = reserveBytes;
        _osMappingBytes = (reserveBytes + (granularity - 1UL)) & ~(granularity - 1UL);

        if (OperatingSystem.IsWindows())
        {
            IntPtr reserved = VirtualAlloc(IntPtr.Zero, (nuint)_osMappingBytes, MEM_RESERVE, PAGE_NOACCESS);
            if (reserved != IntPtr.Zero)
            {
                _reservedBase = reserved;
                _backingMode = BackingMode.WindowsVirtualAlloc;
                return;
            }
        }
        else if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS() || OperatingSystem.IsFreeBSD())
        {
            int mapAnon = (OperatingSystem.IsMacOS() || OperatingSystem.IsFreeBSD())
                ? MAP_ANON_BSD
                : MAP_ANONYMOUS_LINUX;

            IntPtr mapped = mmap(IntPtr.Zero, (nuint)_osMappingBytes, PROT_NONE, MAP_PRIVATE | mapAnon, -1, 0);
            if (mapped != new IntPtr(-1) && mapped != IntPtr.Zero)
            {
                _reservedBase = mapped;
                _backingMode = BackingMode.UnixMmap;
                return;
            }
        }

        // Safe non-zeroing fallback if virtual reservation syscall is unavailable
        _reservedBase = (IntPtr)NativeMemory.AlignedAlloc((nuint)_osMappingBytes, 65536);
        _backingMode = BackingMode.HostAlignedFallback;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VirtualMemoryReservoir"/> class backed by an external device or pinned arena.
    /// Never dereferences <paramref name="baseAddress"/> on the CPU.
    /// </summary>
    /// <param name="baseAddress">Base device or pinned pointer of the backing arena.</param>
    /// <param name="byteLength">Total capacity of the external arena in bytes.</param>
    /// <param name="releaseAction">Optional callback invoked with <paramref name="baseAddress"/> upon <see cref="Dispose"/>.</param>
    public VirtualMemoryReservoir(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        if (baseAddress == IntPtr.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(baseAddress), "External arena base address cannot be zero.");
        }

        if (byteLength == 0 || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength), "External arena capacity must be greater than zero and within addressable bounds.");
        }

        _reservedBase = baseAddress;
        _totalReservedBytes = byteLength;
        _osMappingBytes = byteLength;
        _backingMode = BackingMode.ExternalArena;
        _releaseAction = releaseAction;
    }

    /// <summary>
    /// Commits physical pages (or sub-allocates an external arena slice) for a requested block size.
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
            throw new ArgumentOutOfRangeException(
                nameof(byteCount),
                $"Requested size {byteCount} bytes exceeds total reserved capacity of {_totalReservedBytes} bytes.");
        }

        if (byteCount > ulong.MaxValue - 65535UL)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size causes integer overflow when aligning to page boundary.");
        }

        ulong alignedSize = (byteCount + 65535UL) & ~65535UL;

        lock (_lock)
        {
            if (_committedOffset + alignedSize > _totalReservedBytes)
            {
                throw new OutOfMemoryException(
                    $"Virtual memory reservoir exhausted: requested {alignedSize} bytes, committed {_committedOffset} bytes, reserved capacity {_totalReservedBytes} bytes.");
            }

            IntPtr address = _reservedBase + (nint)_committedOffset;
            if (_backingMode == BackingMode.WindowsVirtualAlloc)
            {
                IntPtr result = VirtualAlloc(address, (nuint)alignedSize, MEM_COMMIT, PAGE_READWRITE);
                if (result == IntPtr.Zero)
                {
                    int win32Error = Marshal.GetLastPInvokeError();
                    throw new OutOfMemoryException(
                        $"VirtualAlloc commit failed at address 0x{address:X16} for {alignedSize} bytes. Win32 error code: {win32Error} (0x{win32Error:X8}).");
                }
            }
            else if (_backingMode == BackingMode.UnixMmap)
            {
                int rc = mprotect(address, (nuint)alignedSize, PROT_READ | PROT_WRITE);
                if (rc != 0)
                {
                    int errno = Marshal.GetLastPInvokeError();
                    throw new OutOfMemoryException(
                        $"mprotect(PROT_READ | PROT_WRITE) failed at address 0x{address:X16} for {alignedSize} bytes. errno: {errno}.");
                }
            }

            _committedOffset += alignedSize;
            _committedAllocations[address] = alignedSize;
            return new DevicePointer(address, alignedSize);
        }
    }

    /// <summary>
    /// Decommits physical pages (or releases the external arena slice) associated with a previously allocated pointer.
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

            if (_backingMode == BackingMode.WindowsVirtualAlloc)
            {
                bool freed = VirtualFree(ptr.Address, (nuint)size, MEM_DECOMMIT);
                if (!freed)
                {
                    int win32Error = Marshal.GetLastPInvokeError();
                    throw new InvalidOperationException(
                        $"VirtualFree decommit failed at address 0x{ptr.Address:X16} for {size} bytes. Win32 error code: {win32Error} (0x{win32Error:X8}).");
                }
            }
            else if (_backingMode == BackingMode.UnixMmap)
            {
                _ = madvise(ptr.Address, (nuint)size, MADV_DONTNEED);
                int rc = mprotect(ptr.Address, (nuint)size, PROT_NONE);
                if (rc != 0)
                {
                    int errno = Marshal.GetLastPInvokeError();
                    throw new InvalidOperationException(
                        $"mprotect(PROT_NONE) decommit failed at address 0x{ptr.Address:X16} for {size} bytes. errno: {errno}.");
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
                    switch (_backingMode)
                    {
                        case BackingMode.WindowsVirtualAlloc:
                            _ = VirtualFree(basePtr, 0, MEM_RELEASE);
                            break;

                        case BackingMode.UnixMmap:
                            _ = munmap(basePtr, (nuint)_osMappingBytes);
                            break;

                        case BackingMode.HostAlignedFallback:
                            NativeMemory.AlignedFree((void*)basePtr);
                            break;

                        case BackingMode.ExternalArena:
                            _releaseAction?.Invoke(basePtr);
                            break;
                    }
                }
            }
        }
    }
}

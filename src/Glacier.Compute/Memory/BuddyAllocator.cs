// <copyright file="BuddyAllocator.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Memory;

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

/// <summary>
/// Tier 2: Buddy Power-of-Two Sub-Allocator (128 KB – 32 MB).
/// Manages intermediate activation buffers, gradient tensors, and convolution maps over either caller-supplied
/// GPU/pinned arenas (e.g. <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>D3D12</c>) or unmanaged host arenas.
/// Guarantees zero external fragmentation by recursively merging companion buddies on free without touching arena memory on the CPU.
/// </summary>
public sealed class BuddyAllocator : IDisposable
{
    /// <summary>Minimum block size: 128 KB.</summary>
    public const ulong MinBlockSize = 128 * 1024; // 131,072 bytes (2^17)

    /// <summary>Maximum block size: 32 MB.</summary>
    public const ulong MaxBlockSize = 32 * 1024 * 1024; // 33,554,432 bytes (2^25)

    /// <summary>Number of buddy orders (Order 0: 128KB, ..., Order 8: 32MB).</summary>
    public const int OrderCount = 9;

    private readonly object _lock = new();
    private IntPtr _arenaBase;
    private readonly ulong _arenaSize;
    private readonly bool _ownsHostMemory;
    private readonly Action<IntPtr>? _releaseAction;
    private readonly HashSet<ulong>[] _freeLists;
    private readonly Dictionary<IntPtr, (ulong Offset, int Order)> _allocatedBlocks;
    private ulong _allocatedBytes;
    private bool _disposed;

    /// <summary>Gets the base native or device address of the buddy arena.</summary>
    public IntPtr BaseAddress
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _arenaBase;
        }
    }

    /// <summary>Gets the total size of the buddy arena in bytes.</summary>
    public ulong TotalCapacity
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _arenaSize;
        }
    }

    /// <summary>Gets the currently allocated memory in bytes.</summary>
    public ulong AllocatedBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            lock (_lock)
            {
                return _allocatedBytes;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BuddyAllocator"/> class backed by an aligned unmanaged host memory arena.
    /// Does not eagerly fault physical pages across the arena at construction time.
    /// </summary>
    /// <param name="arenaSizeBytes">Total host arena capacity in bytes (aligned up to 128 KB; default 64 MB).</param>
    public unsafe BuddyAllocator(ulong arenaSizeBytes = MaxBlockSize * 2)
    {
        if (arenaSizeBytes == 0 || arenaSizeBytes > (ulong)nint.MaxValue - (MinBlockSize - 1))
        {
            throw new ArgumentOutOfRangeException(nameof(arenaSizeBytes), "Arena size must be greater than zero and within addressable memory range.");
        }

        ulong alignedArena = (arenaSizeBytes + MinBlockSize - 1) & ~(MinBlockSize - 1);
        _arenaSize = alignedArena;
        _arenaBase = (IntPtr)NativeMemory.AlignedAlloc((nuint)_arenaSize, 4096);
        _ownsHostMemory = true;

        _freeLists = CreateFreeLists(_arenaSize);
        _allocatedBlocks = new Dictionary<IntPtr, (ulong Offset, int Order)>();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BuddyAllocator"/> class backed by an external device or pinned arena
    /// (such as CUDA <c>cuMemAlloc</c> / <c>cuMemHostAlloc</c> or Direct3D 12 / Vulkan memory).
    /// Never dereferences <paramref name="baseAddress"/> on the CPU.
    /// </summary>
    /// <param name="baseAddress">Base device or pinned pointer of the backing arena.</param>
    /// <param name="byteLength">Total capacity of the external arena in bytes (at least 128 KB).</param>
    /// <param name="releaseAction">Optional callback invoked with <paramref name="baseAddress"/> upon <see cref="Dispose"/>.</param>
    public BuddyAllocator(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        if (baseAddress == IntPtr.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(baseAddress), "External arena base address cannot be zero.");
        }

        if (byteLength < MinBlockSize || byteLength > (ulong)nint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength), $"External buddy arena must be at least {MinBlockSize} bytes and within addressable bounds.");
        }

        _arenaBase = baseAddress;
        _arenaSize = (byteLength / MinBlockSize) * MinBlockSize;
        _ownsHostMemory = false;
        _releaseAction = releaseAction;

        _freeLists = CreateFreeLists(_arenaSize);
        _allocatedBlocks = new Dictionary<IntPtr, (ulong Offset, int Order)>();
    }

    private static HashSet<ulong>[] CreateFreeLists(ulong arenaSize)
    {
        var freeLists = new HashSet<ulong>[OrderCount];
        for (int i = 0; i < OrderCount; i++)
        {
            freeLists[i] = new HashSet<ulong>();
        }

        // Decompose arenaSize into aligned power-of-two buddy blocks from largest order down to Order 0
        ulong offset = 0;
        ulong remaining = arenaSize;
        for (int order = OrderCount - 1; order >= 0; order--)
        {
            ulong blockSize = GetBlockSize(order);
            while (remaining >= blockSize)
            {
                freeLists[order].Add(offset);
                offset += blockSize;
                remaining -= blockSize;
            }
        }

        return freeLists;
    }

    /// <summary>
    /// Calculates the buddy order required to satisfy a requested byte size.
    /// </summary>
    public static int GetOrderForSize(ulong byteCount)
    {
        if (byteCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size must be greater than zero.");
        }

        if (byteCount <= MinBlockSize)
        {
            return 0;
        }

        if (byteCount > MaxBlockSize)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size exceeds maximum buddy block size of 32 MB.");
        }

        ulong units = (byteCount + MinBlockSize - 1) / MinBlockSize;
        if (BitOperations.IsPow2(units))
        {
            return BitOperations.Log2(units);
        }

        return BitOperations.Log2(BitOperations.RoundUpToPowerOf2(units));
    }

    /// <summary>
    /// Returns the block byte size for a given order.
    /// </summary>
    public static ulong GetBlockSize(int order)
    {
        if (order < 0 || order >= OrderCount)
        {
            throw new ArgumentOutOfRangeException(nameof(order), order, $"Buddy order must be between 0 and {OrderCount - 1}.");
        }

        return MinBlockSize << order;
    }

    /// <summary>
    /// Sub-allocates a contiguous block of arena memory for the given byte size.
    /// </summary>
    public DevicePointer Allocate(ulong byteCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (byteCount == 0 || byteCount > MaxBlockSize)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size must be between 1 byte and 32 MB.");
        }

        int targetOrder = GetOrderForSize(byteCount);
        ulong allocatedSize = GetBlockSize(targetOrder);

        lock (_lock)
        {
            int currentOrder = targetOrder;
            while (currentOrder < OrderCount && _freeLists[currentOrder].Count == 0)
            {
                currentOrder++;
            }

            if (currentOrder >= OrderCount)
            {
                throw new OutOfMemoryException(
                    $"BuddyAllocator arena exhausted. Cannot satisfy {byteCount} bytes (Order {targetOrder}). Total capacity: {_arenaSize} bytes, Allocated: {_allocatedBytes} bytes, Active blocks: {_allocatedBlocks.Count}.");
            }

            var enumerator = _freeLists[currentOrder].GetEnumerator();
            enumerator.MoveNext();
            ulong blockOffset = enumerator.Current;
            _freeLists[currentOrder].Remove(blockOffset);

            while (currentOrder > targetOrder)
            {
                currentOrder--;
                ulong childSize = GetBlockSize(currentOrder);
                ulong buddyOffset = blockOffset + childSize;
                _freeLists[currentOrder].Add(buddyOffset);
            }

            IntPtr address = _arenaBase + (nint)blockOffset;
            _allocatedBlocks[address] = (blockOffset, targetOrder);
            _allocatedBytes += allocatedSize;

            return new DevicePointer(address, allocatedSize);
        }
    }

    /// <summary>
    /// Frees a previously allocated buddy block, recursively coalescing companion buddies.
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
            if (!_allocatedBlocks.Remove(ptr.Address, out var metadata))
            {
                return false;
            }

            ulong blockOffset = metadata.Offset;
            int order = metadata.Order;
            ulong allocatedSize = GetBlockSize(order);
            _allocatedBytes -= allocatedSize;

            while (order < OrderCount - 1)
            {
                ulong blockSize = GetBlockSize(order);
                ulong parentSize = blockSize << 1;
                ulong parentStart = blockOffset & ~(parentSize - 1UL);
                if (parentStart + parentSize > _arenaSize)
                {
                    break;
                }

                ulong buddyOffset = blockOffset ^ blockSize;
                if (_freeLists[order].Remove(buddyOffset))
                {
                    blockOffset = parentStart;
                    order++;
                }
                else
                {
                    break;
                }
            }

            _freeLists[order].Add(blockOffset);
            return true;
        }
    }

    /// <summary>
    /// Determines whether the given pointer belongs to this buddy arena.
    /// </summary>
    public bool Contains(IntPtr address)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (address == IntPtr.Zero || _arenaBase == IntPtr.Zero)
        {
            return false;
        }

        nint diff = address - _arenaBase;
        return diff >= 0 && diff < (nint)_arenaSize;
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
                _allocatedBlocks.Clear();
                _allocatedBytes = 0;
                for (int i = 0; i < OrderCount; i++)
                {
                    _freeLists[i].Clear();
                }

                IntPtr basePtr = _arenaBase;
                _arenaBase = IntPtr.Zero;
                if (basePtr != IntPtr.Zero)
                {
                    if (_ownsHostMemory)
                    {
                        NativeMemory.AlignedFree((void*)basePtr);
                    }
                    else
                    {
                        _releaseAction?.Invoke(basePtr);
                    }
                }
            }
        }
    }
}

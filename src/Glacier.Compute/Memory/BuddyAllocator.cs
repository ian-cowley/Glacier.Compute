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
/// Tier 2: Buddy Power-of-Two Allocator (128 KB – 32 MB).
/// Manages intermediate activation buffers, gradient tensors, and convolution maps.
/// Guarantees zero external fragmentation by recursively merging companion buddies on free.
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
    private readonly IntPtr _arenaBase;
    private readonly ulong _arenaSize;
    private readonly HashSet<ulong>[] _freeLists;
    private readonly Dictionary<IntPtr, (ulong Offset, int Order)> _allocatedBlocks;
    private bool _disposed;

    /// <summary>Gets the total size of the buddy arena in bytes.</summary>
    public ulong TotalCapacity => _arenaSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="BuddyAllocator"/> class.
    /// </summary>
    /// <param name="arenaSizeBytes">Total capacity in bytes (must be a multiple of MaxBlockSize, default 64 MB).</param>
    public unsafe BuddyAllocator(ulong arenaSizeBytes = MaxBlockSize * 2)
    {
        if (arenaSizeBytes < MaxBlockSize || (arenaSizeBytes % MaxBlockSize) != 0)
        {
            arenaSizeBytes = MaxBlockSize * 2;
        }

        _arenaSize = arenaSizeBytes;
        _arenaBase = (IntPtr)NativeMemory.AlignedAlloc((nuint)_arenaSize, 4096);
        NativeMemory.Clear((void*)_arenaBase, (nuint)_arenaSize);

        _freeLists = new HashSet<ulong>[OrderCount];
        for (int i = 0; i < OrderCount; i++)
        {
            _freeLists[i] = new HashSet<ulong>();
        }

        _allocatedBlocks = new Dictionary<IntPtr, (ulong Offset, int Order)>();

        // Populate top-level 32 MB blocks into the highest order free list
        int rootOrder = OrderCount - 1;
        ulong rootBlockCount = _arenaSize / MaxBlockSize;
        for (ulong i = 0; i < rootBlockCount; i++)
        {
            _freeLists[rootOrder].Add(i * MaxBlockSize);
        }
    }

    /// <summary>
    /// Calculates the buddy order required to satisfy a requested byte size.
    /// </summary>
    public static int GetOrderForSize(ulong byteCount)
    {
        if (byteCount <= MinBlockSize)
        {
            return 0;
        }

        if (byteCount > MaxBlockSize)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Requested size exceeds maximum buddy block size of 32 MB.");
        }

        // byteCount = MinBlockSize * 2^order
        // 2^order >= byteCount / MinBlockSize
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
    public static ulong GetBlockSize(int order) => MinBlockSize << order;

    /// <summary>
    /// Allocates a contiguous block of device memory for the given byte size.
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
            // Find lowest order >= targetOrder that has a free block
            int currentOrder = targetOrder;
            while (currentOrder < OrderCount && _freeLists[currentOrder].Count == 0)
            {
                currentOrder++;
            }

            if (currentOrder >= OrderCount)
            {
                throw new OutOfMemoryException($"BuddyAllocator arena exhausted. Cannot satisfy {byteCount} bytes (Order {targetOrder}).");
            }

            // Pop a block from currentOrder
            var enumerator = _freeLists[currentOrder].GetEnumerator();
            enumerator.MoveNext();
            ulong blockOffset = enumerator.Current;
            _freeLists[currentOrder].Remove(blockOffset);

            // Split downwards to targetOrder
            while (currentOrder > targetOrder)
            {
                currentOrder--;
                ulong childSize = GetBlockSize(currentOrder);
                ulong buddyOffset = blockOffset + childSize;
                _freeLists[currentOrder].Add(buddyOffset);
            }

            IntPtr address = _arenaBase + (nint)blockOffset;
            _allocatedBlocks[address] = (blockOffset, targetOrder);

            return new DevicePointer(address, allocatedSize);
        }
    }

    /// <summary>
    /// Frees a previously allocated buddy block, recursively coalescing companion buddies.
    /// </summary>
    public bool Free(DevicePointer ptr)
    {
        if (_disposed || ptr.IsNull)
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

            // Coalesce buddies recursively
            while (order < OrderCount - 1)
            {
                ulong blockSize = GetBlockSize(order);
                ulong buddyOffset = blockOffset ^ blockSize;

                // Check if buddy block is free
                if (_freeLists[order].Remove(buddyOffset))
                {
                    // Buddy is free! Coalesce into parent block at min(blockOffset, buddyOffset)
                    blockOffset = Math.Min(blockOffset, buddyOffset);
                    order++;
                }
                else
                {
                    // Buddy is not free, stop coalescing
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
        nint diff = address - _arenaBase;
        return diff >= 0 && diff < (nint)_arenaSize;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            lock (_lock)
            {
                _allocatedBlocks.Clear();
                for (int i = 0; i < OrderCount; i++)
                {
                    _freeLists[i].Clear();
                }
                if (_arenaBase != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree((void*)_arenaBase);
                }
            }
        }
    }
}

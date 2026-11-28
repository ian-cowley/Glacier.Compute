// <copyright file="RingSocketTransport.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// High-performance TCP socket transport for distributed ring AllReduce.
/// Implements pooled coalesced framing with configurable <see cref="MaxFrameBytes"/> validation over TCP sockets.
/// </summary>
public sealed class RingSocketTransport : IRingTransport
{
    /// <summary>Default maximum allowed frame payload size in bytes (64 MB).</summary>
    public const int DefaultMaxFrameBytes = 64 * 1024 * 1024;

    private readonly int _rank;
    private readonly int _worldSize;
    private readonly Socket _successorSocket;
    private readonly Socket _predecessorSocket;
    private int _disposed;

    /// <inheritdoc/>
    public int Rank => _rank;

    /// <inheritdoc/>
    public int WorldSize => _worldSize;

    /// <summary>Gets the maximum allowed frame payload size in bytes.</summary>
    public int MaxFrameBytes { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="RingSocketTransport"/> class.
    /// </summary>
    public RingSocketTransport(
        int rank,
        int worldSize,
        Socket successorSocket,
        Socket predecessorSocket,
        int maxFrameBytes = DefaultMaxFrameBytes)
    {
        if (worldSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(worldSize), worldSize, "WorldSize must be greater than zero.");
        }

        if (rank < 0 || rank >= worldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"Rank must be in [0, {worldSize - 1}].");
        }

        if (maxFrameBytes < sizeof(float))
        {
            throw new ArgumentOutOfRangeException(nameof(maxFrameBytes), maxFrameBytes, "MaxFrameBytes must be at least 4 bytes.");
        }

        _rank = rank;
        _worldSize = worldSize;
        _successorSocket = successorSocket ?? throw new ArgumentNullException(nameof(successorSocket));
        _predecessorSocket = predecessorSocket ?? throw new ArgumentNullException(nameof(predecessorSocket));
        MaxFrameBytes = maxFrameBytes;

        ConfigureSocket(_successorSocket);
        ConfigureSocket(_predecessorSocket);
    }

    private static void ConfigureSocket(Socket s)
    {
        s.NoDelay = true;
        s.SendBufferSize = 1024 * 1024;
        s.ReceiveBufferSize = 1024 * 1024;
    }

    /// <inheritdoc/>
    public async ValueTask SendToSuccessorAsync(ReadOnlyMemory<float> data, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        long byteLengthLong = (long)data.Length * sizeof(float);
        if (byteLengthLong > MaxFrameBytes)
        {
            throw new ArgumentOutOfRangeException(
                nameof(data),
                $"Frame payload size {byteLengthLong} bytes exceeds MaxFrameBytes ({MaxFrameBytes} bytes).");
        }

        int byteLength = (int)byteLengthLong;
        byte[] rented = ArrayPool<byte>.Shared.Rent(4 + byteLength);
        try
        {
            BinaryPrimitives.WriteInt32LittleEndian(rented.AsSpan(0, 4), byteLength);
            if (byteLength > 0)
            {
                MemoryMarshal.AsBytes(data.Span).CopyTo(rented.AsSpan(4, byteLength));
            }

            await SendAllAsync(_successorSocket, rented.AsMemory(0, 4 + byteLength), ct).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <inheritdoc/>
    public async ValueTask ReceiveFromPredecessorAsync(Memory<float> buffer, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        int byteLength;
        byte[] header = ArrayPool<byte>.Shared.Rent(4);
        try
        {
            await ReceiveExactAsync(_predecessorSocket, header.AsMemory(0, 4), ct).ConfigureAwait(false);
            byteLength = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(0, 4));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(header);
        }

        if (byteLength < 0 || byteLength > MaxFrameBytes || (byteLength % sizeof(float)) != 0)
        {
            throw new InvalidDataException(
                $"Invalid ring transport frame length {byteLength} bytes (allowed 0..{MaxFrameBytes}, aligned to {sizeof(float)} bytes).");
        }

        if (byteLength == 0)
        {
            return;
        }

        byte[] rawBytes = ArrayPool<byte>.Shared.Rent(byteLength);
        try
        {
            await ReceiveExactAsync(_predecessorSocket, rawBytes.AsMemory(0, byteLength), ct).ConfigureAwait(false);

            int floatCount = byteLength / sizeof(float);
            int targetFloats = Math.Min(floatCount, buffer.Length);
            ReadOnlySpan<float> receivedFloats = MemoryMarshal.Cast<byte, float>(rawBytes.AsSpan(0, targetFloats * sizeof(float)));
            receivedFloats.CopyTo(buffer.Span);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rawBytes);
        }
    }

    private static async ValueTask SendAllAsync(Socket socket, ReadOnlyMemory<byte> buffer, CancellationToken ct)
    {
        int sent = 0;
        while (sent < buffer.Length)
        {
            int bytesSent = await socket.SendAsync(buffer.Slice(sent), SocketFlags.None, ct).ConfigureAwait(false);
            if (bytesSent == 0)
            {
                throw new SocketException((int)SocketError.ConnectionReset);
            }
            sent += bytesSent;
        }
    }

    private static async ValueTask ReceiveExactAsync(Socket socket, Memory<byte> buffer, CancellationToken ct)
    {
        int received = 0;
        while (received < buffer.Length)
        {
            int bytesRecv = await socket.ReceiveAsync(buffer.Slice(received), SocketFlags.None, ct).ConfigureAwait(false);
            if (bytesRecv == 0)
            {
                throw new SocketException((int)SocketError.ConnectionReset);
            }
            received += bytesRecv;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            try { _successorSocket.Dispose(); }
            catch (Exception ex) when (ex is ObjectDisposedException or System.Net.Sockets.SocketException)
            {
                System.Diagnostics.Trace.TraceWarning($"[RingSocketTransport] Successor socket dispose warning: {ex.Message}");
            }

            try { _predecessorSocket.Dispose(); }
            catch (Exception ex) when (ex is ObjectDisposedException or System.Net.Sockets.SocketException)
            {
                System.Diagnostics.Trace.TraceWarning($"[RingSocketTransport] Predecessor socket dispose warning: {ex.Message}");
            }
        }
    }
}

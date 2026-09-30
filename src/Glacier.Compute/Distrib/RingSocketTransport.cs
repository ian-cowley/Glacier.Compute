// <copyright file="RingSocketTransport.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// High-performance TCP socket transport for distributed ring AllReduce.
/// Implements zero-copy streaming over non-blocking TCP sockets.
/// </summary>
public sealed class RingSocketTransport : IRingTransport
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly Socket _successorSocket;
    private readonly Socket _predecessorSocket;
    private bool _disposed;

    /// <inheritdoc/>
    public int Rank => _rank;

    /// <inheritdoc/>
    public int WorldSize => _worldSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="RingSocketTransport"/> class.
    /// </summary>
    public RingSocketTransport(int rank, int worldSize, Socket successorSocket, Socket predecessorSocket)
    {
        _rank = rank;
        _worldSize = worldSize;
        _successorSocket = successorSocket ?? throw new ArgumentNullException(nameof(successorSocket));
        _predecessorSocket = predecessorSocket ?? throw new ArgumentNullException(nameof(predecessorSocket));

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
        ObjectDisposedException.ThrowIf(_disposed, this);

        ReadOnlyMemory<byte> byteData = MemoryMarshal.AsBytes(data.Span).ToArray();
        // Send 4-byte byteCount prefix
        byte[] header = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(header, byteData.Length);

        await _successorSocket.SendAsync(header, SocketFlags.None, ct).ConfigureAwait(false);
        await SendAllAsync(_successorSocket, byteData, ct).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async ValueTask ReceiveFromPredecessorAsync(Memory<float> buffer, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        byte[] header = new byte[4];
        await ReceiveExactAsync(_predecessorSocket, header, ct).ConfigureAwait(false);
        int byteLength = BinaryPrimitives.ReadInt32LittleEndian(header);

        int floatCount = byteLength / sizeof(float);
        int targetFloats = Math.Min(floatCount, buffer.Length);

        byte[] rawBytes = new byte[byteLength];
        await ReceiveExactAsync(_predecessorSocket, rawBytes, ct).ConfigureAwait(false);

        ReadOnlySpan<float> receivedFloats = MemoryMarshal.Cast<byte, float>(rawBytes.AsSpan(0, targetFloats * sizeof(float)));
        receivedFloats.CopyTo(buffer.Span);
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
        if (!_disposed)
        {
            _disposed = true;
            try { _successorSocket.Dispose(); } catch { }
            try { _predecessorSocket.Dispose(); } catch { }
        }
    }
}

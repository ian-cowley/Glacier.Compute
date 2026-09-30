// <copyright file="InMemoryRingTransport.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

/// <summary>
/// In-memory zero-copy ring transport for local multi-rank simulation and microbenchmarks.
/// </summary>
public sealed class InMemoryRingTransport : IRingTransport
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly Channel<float[]> _sendChannel;
    private readonly Channel<float[]> _recvChannel;
    private bool _disposed;

    /// <inheritdoc/>
    public int Rank => _rank;

    /// <inheritdoc/>
    public int WorldSize => _worldSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="InMemoryRingTransport"/> class.
    /// </summary>
    public InMemoryRingTransport(
        int rank,
        int worldSize,
        Channel<float[]> sendToSuccessor,
        Channel<float[]> recvFromPredecessor)
    {
        _rank = rank;
        _worldSize = worldSize;
        _sendChannel = sendToSuccessor;
        _recvChannel = recvFromPredecessor;
    }

    /// <summary>
    /// Creates an in-memory cluster mesh of <paramref name="worldSize"/> interconnected ring transports.
    /// </summary>
    public static InMemoryRingTransport[] CreateCluster(int worldSize)
    {
        if (worldSize <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(worldSize), "World size must be at least 2.");
        }

        // Channel from rank i to rank (i + 1) % worldSize
        var channels = new Channel<float[]>[worldSize];
        for (int i = 0; i < worldSize; i++)
        {
            channels[i] = Channel.CreateBounded<float[]>(new BoundedChannelOptions(16)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });
        }

        var cluster = new InMemoryRingTransport[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            int succ = r;
            int pred = (r - 1 + worldSize) % worldSize;
            cluster[r] = new InMemoryRingTransport(r, worldSize, channels[succ], channels[pred]);
        }

        return cluster;
    }

    /// <inheritdoc/>
    public async ValueTask SendToSuccessorAsync(ReadOnlyMemory<float> data, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        float[] copy = data.ToArray();
        await _sendChannel.Writer.WriteAsync(copy, ct).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async ValueTask ReceiveFromPredecessorAsync(Memory<float> buffer, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        float[] item = await _recvChannel.Reader.ReadAsync(ct).ConfigureAwait(false);
        item.AsSpan(0, Math.Min(item.Length, buffer.Length)).CopyTo(buffer.Span);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _sendChannel.Writer.TryComplete();
        }
    }
}

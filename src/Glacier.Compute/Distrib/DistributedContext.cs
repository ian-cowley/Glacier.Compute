// <copyright file="DistributedContext.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// High-performance pure C# implementation of <see cref="IDistributedContext"/> executing
/// Ring AllReduce with SIMD vector accumulation across cluster nodes.
/// </summary>
public sealed class DistributedContext : IDistributedContext
{
    private readonly IRingTransport _transport;
    private bool _disposed;

    /// <inheritdoc/>
    public int Rank => _transport.Rank;

    /// <inheritdoc/>
    public int WorldSize => _transport.WorldSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="DistributedContext"/> class.
    /// </summary>
    public DistributedContext(IRingTransport transport)
    {
        _transport = transport ?? throw new ArgumentNullException(nameof(transport));
    }

    /// <inheritdoc/>
    public async ValueTask AllReduceAsync(Memory<float> buffer, ReductionOp op, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (buffer.Length == 0 || WorldSize <= 1)
        {
            return;
        }

        int n = WorldSize;
        int totalElements = buffer.Length;

        // Partition tensor buffer into N chunks
        int baseChunk = totalElements / n;
        int rem = totalElements % n;

        int[] offsets = new int[n];
        int[] lengths = new int[n];

        int currOffset = 0;
        for (int i = 0; i < n; i++)
        {
            offsets[i] = currOffset;
            lengths[i] = baseChunk + (i < rem ? 1 : 0);
            currOffset += lengths[i];
        }

        // Allocate scratch buffer for receiving chunks
        int maxChunkLength = baseChunk + (rem > 0 ? 1 : 0);
        float[] recvScratch = new float[maxChunkLength];

        // -------------------------------------------------------------------
        // Phase 1: Scatter-Reduce (N - 1 steps)
        // -------------------------------------------------------------------
        for (int step = 0; step < n - 1; step++)
        {
            int sendIdx = (Rank - step + n) % n;
            int recvIdx = (Rank - step - 1 + n) % n;

            var sendMem = buffer.Slice(offsets[sendIdx], lengths[sendIdx]);
            var recvMem = recvScratch.AsMemory(0, lengths[recvIdx]);

            // Transmit to successor and receive from predecessor concurrently
            var sendTask = _transport.SendToSuccessorAsync(sendMem, ct);
            var recvTask = _transport.ReceiveFromPredecessorAsync(recvMem, ct);

            await sendTask.ConfigureAwait(false);
            await recvTask.ConfigureAwait(false);

            // In-place vector accumulation with SIMD
            var localSlice = buffer.Span.Slice(offsets[recvIdx], lengths[recvIdx]);
            SimdAccumulator.Accumulate(localSlice, recvMem.Span, op);
        }

        // -------------------------------------------------------------------
        // Phase 2: AllGather (N - 1 steps)
        // -------------------------------------------------------------------
        for (int step = 0; step < n - 1; step++)
        {
            int sendIdx = (Rank - step + 1 + n) % n;
            int recvIdx = (Rank - step + n) % n;

            var sendMem = buffer.Slice(offsets[sendIdx], lengths[sendIdx]);
            var recvMem = buffer.Slice(offsets[recvIdx], lengths[recvIdx]);

            var sendTask = _transport.SendToSuccessorAsync(sendMem, ct);
            var recvTask = _transport.ReceiveFromPredecessorAsync(recvMem, ct);

            await sendTask.ConfigureAwait(false);
            await recvTask.ConfigureAwait(false);
        }

        // If Average operation, scale the reduced sum by 1 / WorldSize
        if (op == ReductionOp.Average)
        {
            SimdAccumulator.Scale(buffer.Span, 1.0f / n);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _transport.Dispose();
        }
    }
}

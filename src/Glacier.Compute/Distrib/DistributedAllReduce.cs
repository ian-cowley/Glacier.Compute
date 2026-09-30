// <copyright file="DistributedAllReduce.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Distributed AllReduce coordinator and factory for cluster communications.
/// Replaces NVIDIA NCCL with a pure C# .NET 10 cross-platform ring reduction engine.
/// </summary>
public static class DistributedAllReduce
{
    /// <summary>
    /// Creates a local in-memory simulated cluster of <paramref name="worldSize"/> interconnected nodes.
    /// </summary>
    /// <param name="worldSize">Total number of cluster nodes in the ring.</param>
    /// <returns>An array of <see cref="IDistributedContext"/> instances, one per rank.</returns>
    public static IDistributedContext[] CreateLocalCluster(int worldSize)
    {
        var transports = InMemoryRingTransport.CreateCluster(worldSize);
        var contexts = new IDistributedContext[worldSize];
        for (int i = 0; i < worldSize; i++)
        {
            contexts[i] = new DistributedContext(transports[i]);
        }
        return contexts;
    }

    /// <summary>
    /// Creates a distributed socket context connected to its ring neighbors.
    /// </summary>
    public static IDistributedContext CreateSocketContext(int rank, int worldSize, Socket successorSocket, Socket predecessorSocket)
    {
        var transport = new RingSocketTransport(rank, worldSize, successorSocket, predecessorSocket);
        return new DistributedContext(transport);
    }

    /// <summary>
    /// Simultaneously executes an AllReduce operation across all cluster ranks on their respective buffers.
    /// </summary>
    public static async ValueTask ExecuteClusterAsync(
        IDistributedContext[] contexts,
        Memory<float>[] buffers,
        ReductionOp op = ReductionOp.Sum,
        CancellationToken ct = default)
    {
        if (contexts.Length != buffers.Length)
        {
            throw new ArgumentException("Number of contexts and buffers must match.");
        }

        var tasks = new Task[contexts.Length];
        for (int i = 0; i < contexts.Length; i++)
        {
            tasks[i] = contexts[i].AllReduceAsync(buffers[i], op, ct).AsTask();
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
    }
}

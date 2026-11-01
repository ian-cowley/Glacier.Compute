// <copyright file="SocketRingAllReduceTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.DistribTests;

using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Distrib;
using Xunit;

public class SocketRingAllReduceTests
{
    [Fact]
    public async Task SocketRingAllReduce_3NodesOverTcp_SynchronizesTensors()
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        CancellationToken ct = cts.Token;

        const int worldSize = 3;
        const int tensorSize = 256;

        var listeners = new TcpListener[worldSize];
        var ports = new int[worldSize];
        var succClients = new Socket?[worldSize];
        var predSockets = new Socket?[worldSize];
        var contexts = new IDistributedContext?[worldSize];

        try
        {
            for (int i = 0; i < worldSize; i++)
            {
                listeners[i] = new TcpListener(IPAddress.Loopback, 0);
                listeners[i].Start();
                ports[i] = ((IPEndPoint)listeners[i].LocalEndpoint).Port;
            }

            var acceptTasks = new Task<Socket>[worldSize];
            for (int i = 0; i < worldSize; i++)
            {
                acceptTasks[i] = listeners[i].AcceptSocketAsync(ct).AsTask();
            }

            for (int i = 0; i < worldSize; i++)
            {
                int succPort = ports[(i + 1) % worldSize];
                var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                await client.ConnectAsync(IPAddress.Loopback, succPort, ct);
                succClients[i] = client;
            }

            for (int i = 0; i < worldSize; i++)
            {
                predSockets[i] = await acceptTasks[i].WaitAsync(ct);
                listeners[i].Stop();
            }

            var buffers = new Memory<float>[worldSize];
            var arrays = new float[worldSize][];

            for (int r = 0; r < worldSize; r++)
            {
                contexts[r] = DistributedAllReduce.CreateSocketContext(r, worldSize, succClients[r]!, predSockets[r]!);
                arrays[r] = new float[tensorSize];
                for (int i = 0; i < tensorSize; i++)
                {
                    arrays[r][i] = 1.0f * (r + 1); // Rank 0: 1, Rank 1: 2, Rank 2: 3 -> sum = 6
                }
                buffers[r] = arrays[r];
            }

            await DistributedAllReduce.ExecuteClusterAsync(contexts!, buffers, ReductionOp.Sum, ct);

            for (int r = 0; r < worldSize; r++)
            {
                for (int i = 0; i < tensorSize; i++)
                {
                    Assert.Equal(6.0f, arrays[r][i], 3);
                }
            }
        }
        finally
        {
            for (int r = 0; r < worldSize; r++)
            {
                try { listeners[r]?.Stop(); } catch { }
                contexts[r]?.Dispose();
                succClients[r]?.Dispose();
                predSockets[r]?.Dispose();
            }
        }
    }
}

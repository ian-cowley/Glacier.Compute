// <copyright file="SocketRingAllReduceTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.DistribTests;

using System;
using System.Buffers.Binary;
using System.IO;
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

    [Fact]
    public void RingSocketTransport_ValidatesConstructorArguments()
    {
        using var s1 = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        using var s2 = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        Assert.Throws<ArgumentOutOfRangeException>(() => new RingSocketTransport(0, 0, s1, s2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RingSocketTransport(-1, 2, s1, s2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RingSocketTransport(2, 2, s1, s2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RingSocketTransport(0, 2, s1, s2, maxFrameBytes: 3));
    }

    [Fact]
    public async Task RingSocketTransport_RejectsOversizedSend_AndAdversarialReceiveHeaders_WithoutAllocation()
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        using var senderSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var acceptTask = listener.AcceptSocketAsync(cts.Token).AsTask();
        await senderSocket.ConnectAsync(IPAddress.Loopback, port, cts.Token);
        using var receiverSocket = await acceptTask.WaitAsync(cts.Token);
        listener.Stop();

        const int maxFrameBytes = 64; // max 16 floats
        using var transport = new RingSocketTransport(0, 2, senderSocket, receiverSocket, maxFrameBytes);

        // Send exceeding MaxFrameBytes must fail before sending
        float[] tooBig = new float[17]; // 68 bytes > 64
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
            await transport.SendToSuccessorAsync(tooBig, cts.Token));

        // Adversarial length headers sent raw from peer to receiver
        float[] recvBuffer = new float[16];
        int[] badLengths = { -1, 5, maxFrameBytes + 4, int.MaxValue };
        byte[] header = new byte[4];

        foreach (int badLen in badLengths)
        {
            BinaryPrimitives.WriteInt32LittleEndian(header, badLen);
            await senderSocket.SendAsync(header, SocketFlags.None, cts.Token);

            await Assert.ThrowsAsync<InvalidDataException>(async () =>
                await transport.ReceiveFromPredecessorAsync(recvBuffer, cts.Token));
        }

        // Verify valid chunked send/receive still works after configuring sockets
        float[] valid = { 1.5f, -2.25f, 3.75f, 4.0f };
        await transport.SendToSuccessorAsync(valid, cts.Token);
        float[] received = new float[4];
        await transport.ReceiveFromPredecessorAsync(received, cts.Token);
        Assert.Equal(valid, received);
    }
}

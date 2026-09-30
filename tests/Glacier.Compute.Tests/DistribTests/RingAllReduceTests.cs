// <copyright file="RingAllReduceTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.DistribTests;

using System;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Distrib;
using Xunit;

public class RingAllReduceTests
{
    [Theory]
    [InlineData(2, 64)]
    [InlineData(3, 100)]
    [InlineData(4, 256)]
    [InlineData(8, 1024)]
    public async Task RingAllReduce_Sum_ProducesIdenticalOutputs(int worldSize, int tensorSize)
    {
        var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
        var buffers = new Memory<float>[worldSize];
        var arrays = new float[worldSize][];

        for (int r = 0; r < worldSize; r++)
        {
            arrays[r] = new float[tensorSize];
            for (int i = 0; i < tensorSize; i++)
            {
                arrays[r][i] = (r + 1) * 2.0f; // e.g., rank 0 has 2, rank 1 has 4, etc.
            }
            buffers[r] = arrays[r];
        }

        // Expected sum: sum of (r + 1) * 2.0f for r=0..worldSize-1
        float expectedSum = 0;
        for (int r = 0; r < worldSize; r++)
        {
            expectedSum += (r + 1) * 2.0f;
        }

        await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, ReductionOp.Sum);

        // All nodes must hold the exact same accumulated sum
        for (int r = 0; r < worldSize; r++)
        {
            for (int i = 0; i < tensorSize; i++)
            {
                Assert.Equal(expectedSum, arrays[r][i], 3);
            }
        }
    }

    [Fact]
    public async Task RingAllReduce_Average_ProducesMean()
    {
        const int worldSize = 4;
        const int tensorSize = 128;

        var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
        var buffers = new Memory<float>[worldSize];
        var arrays = new float[worldSize][];

        for (int r = 0; r < worldSize; r++)
        {
            arrays[r] = new float[tensorSize];
            for (int i = 0; i < tensorSize; i++)
            {
                arrays[r][i] = (r + 1) * 10.0f; // 10, 20, 30, 40
            }
            buffers[r] = arrays[r];
        }

        // Expected mean: (10 + 20 + 30 + 40) / 4 = 25.0f
        const float expectedMean = 25.0f;

        await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, ReductionOp.Average);

        for (int r = 0; r < worldSize; r++)
        {
            for (int i = 0; i < tensorSize; i++)
            {
                Assert.Equal(expectedMean, arrays[r][i], 3);
            }
        }
    }

    [Fact]
    public async Task RingAllReduce_MinAndMax_Parity()
    {
        const int worldSize = 3;
        const int tensorSize = 60;

        // Test Min
        {
            var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
            var buffers = new Memory<float>[worldSize];
            var arrays = new float[worldSize][];

            for (int r = 0; r < worldSize; r++)
            {
                arrays[r] = new float[tensorSize];
                for (int i = 0; i < tensorSize; i++)
                {
                    arrays[r][i] = (r + 1) * 5.0f; // 5, 10, 15
                }
                buffers[r] = arrays[r];
            }

            await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, ReductionOp.Min);

            for (int r = 0; r < worldSize; r++)
            {
                for (int i = 0; i < tensorSize; i++)
                {
                    Assert.Equal(5.0f, arrays[r][i], 3);
                }
            }
        }

        // Test Max
        {
            var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
            var buffers = new Memory<float>[worldSize];
            var arrays = new float[worldSize][];

            for (int r = 0; r < worldSize; r++)
            {
                arrays[r] = new float[tensorSize];
                for (int i = 0; i < tensorSize; i++)
                {
                    arrays[r][i] = (r + 1) * 5.0f; // 5, 10, 15
                }
                buffers[r] = arrays[r];
            }

            await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, ReductionOp.Max);

            for (int r = 0; r < worldSize; r++)
            {
                for (int i = 0; i < tensorSize; i++)
                {
                    Assert.Equal(15.0f, arrays[r][i], 3);
                }
            }
        }
    }
}

// <copyright file="AllReduceBenchmarks.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Benchmarks;

using System;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using Glacier.Compute;
using Glacier.Compute.Distrib;

[MemoryDiagnoser]
public class AllReduceBenchmarks
{
    private float[] _dst = null!;
    private float[] _src = null!;

    private IDistributedContext[] _cluster4 = null!;
    private Memory<float>[] _buffers4 = null!;

    [Params(1024, 65536)]
    public int TensorSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _dst = new float[TensorSize];
        _src = new float[TensorSize];
        Array.Fill(_dst, 1.0f);
        Array.Fill(_src, 2.0f);

        _cluster4 = DistributedAllReduce.CreateLocalCluster(4);
        _buffers4 = new Memory<float>[4];
        for (int i = 0; i < 4; i++)
        {
            _buffers4[i] = new float[TensorSize];
        }
    }

    [Benchmark]
    public void SimdAccumulateSum()
    {
        SimdAccumulator.Accumulate(_dst, _src, ReductionOp.Sum);
    }

    [Benchmark]
    public async Task RingAllReduce_4Nodes()
    {
        await DistributedAllReduce.ExecuteClusterAsync(_cluster4, _buffers4, ReductionOp.Sum);
    }
}

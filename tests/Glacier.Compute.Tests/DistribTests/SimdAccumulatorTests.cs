// <copyright file="SimdAccumulatorTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.DistribTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Distrib;
using Xunit;

public class SimdAccumulatorTests
{
    [Theory]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(33)]
    [InlineData(64)]
    [InlineData(512)]
    [InlineData(1024)]
    public void SimdAccumulator_AccumulateSum_MatchesScalar(int length)
    {
        float[] dst = new float[length];
        float[] src = new float[length];
        float[] expected = new float[length];

        for (int i = 0; i < length; i++)
        {
            dst[i] = i * 1.5f;
            src[i] = (i + 1) * 2.0f;
            expected[i] = dst[i] + src[i];
        }

        SimdAccumulator.Accumulate(dst, src, ReductionOp.Sum);

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(expected[i], dst[i], 4);
        }
    }

    [Theory]
    [InlineData(16)]
    [InlineData(65)]
    [InlineData(256)]
    public void SimdAccumulator_AccumulateMinAndMax_MatchesScalar(int length)
    {
        float[] dstMin = new float[length];
        float[] dstMax = new float[length];
        float[] src = new float[length];
        float[] expectedMin = new float[length];
        float[] expectedMax = new float[length];

        for (int i = 0; i < length; i++)
        {
            dstMin[i] = (float)Math.Sin(i);
            dstMax[i] = (float)Math.Sin(i);
            src[i] = (float)Math.Cos(i);
            expectedMin[i] = Math.Min(dstMin[i], src[i]);
            expectedMax[i] = Math.Max(dstMax[i], src[i]);
        }

        SimdAccumulator.Accumulate(dstMin, src, ReductionOp.Min);
        SimdAccumulator.Accumulate(dstMax, src, ReductionOp.Max);

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(expectedMin[i], dstMin[i], 5);
            Assert.Equal(expectedMax[i], dstMax[i], 5);
        }
    }

    [Fact]
    public void SimdAccumulator_Scale_MultipliesVectorized()
    {
        int length = 512;
        float[] buffer = new float[length];
        float[] expected = new float[length];

        for (int i = 0; i < length; i++)
        {
            buffer[i] = i + 1.0f;
            expected[i] = (i + 1.0f) * 0.25f;
        }

        SimdAccumulator.Scale(buffer, 0.25f);

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(expected[i], buffer[i], 5);
        }
    }
}

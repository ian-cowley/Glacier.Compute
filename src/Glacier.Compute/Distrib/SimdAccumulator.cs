// <copyright file="SimdAccumulator.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// Ultra-fast SIMD vector accumulator leveraging Vector512, Vector256, and Vector128 intrinsics.
/// Delivers peak bandwidth tensor reduction without heap allocations.
/// </summary>
public static class SimdAccumulator
{
    /// <summary>
    /// Accumulates elements from source into destination according to the reduction operator.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Accumulate(Span<float> destination, ReadOnlySpan<float> source, ReductionOp op)
    {
        if (destination.Length != source.Length)
        {
            throw new ArgumentException("Destination and source spans must have identical length.");
        }

        switch (op)
        {
            case ReductionOp.Sum:
            case ReductionOp.Average:
                AccumulateSum(destination, source);
                break;

            case ReductionOp.Min:
                AccumulateMin(destination, source);
                break;

            case ReductionOp.Max:
                AccumulateMax(destination, source);
                break;

            default:
                throw new NotSupportedException($"Unsupported reduction operator: {op}");
        }
    }

    /// <summary>
    /// Scales all elements in the buffer by a constant factor using SIMD multiplication.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scale(Span<float> buffer, float factor)
    {
        int i = 0;
        int length = buffer.Length;

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vFactor = Vector512.Create(factor);
            ref float pBuf = ref MemoryMarshal.GetReference(buffer);
            int vCount = Vector512<float>.Count;

            for (; i <= length - vCount; i += vCount)
            {
                var v = Vector512.LoadUnsafe(ref pBuf, (nuint)i);
                (v * vFactor).StoreUnsafe(ref pBuf, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vFactor = Vector256.Create(factor);
            ref float pBuf = ref MemoryMarshal.GetReference(buffer);
            int vCount = Vector256<float>.Count;

            for (; i <= length - vCount; i += vCount)
            {
                var v = Vector256.LoadUnsafe(ref pBuf, (nuint)i);
                (v * vFactor).StoreUnsafe(ref pBuf, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vFactor = Vector128.Create(factor);
            ref float pBuf = ref MemoryMarshal.GetReference(buffer);
            int vCount = Vector128<float>.Count;

            for (; i <= length - vCount; i += vCount)
            {
                var v = Vector128.LoadUnsafe(ref pBuf, (nuint)i);
                (v * vFactor).StoreUnsafe(ref pBuf, (nuint)i);
            }
        }

        // Remainder
        for (; i < length; i++)
        {
            buffer[i] *= factor;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateSum(Span<float> dst, ReadOnlySpan<float> src)
    {
        int i = 0;
        int length = dst.Length;
        ref float pDst = ref MemoryMarshal.GetReference(dst);
        ref float pSrc = ref MemoryMarshal.GetReference(src);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            int vCount = Vector512<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector512.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector512.LoadUnsafe(ref pSrc, (nuint)i);
                (vd + vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            int vCount = Vector256<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector256.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector256.LoadUnsafe(ref pSrc, (nuint)i);
                (vd + vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            int vCount = Vector128<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector128.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector128.LoadUnsafe(ref pSrc, (nuint)i);
                (vd + vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }

        // Remainder
        for (; i < length; i++)
        {
            dst[i] += src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateMin(Span<float> dst, ReadOnlySpan<float> src)
    {
        int i = 0;
        int length = dst.Length;
        ref float pDst = ref MemoryMarshal.GetReference(dst);
        ref float pSrc = ref MemoryMarshal.GetReference(src);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            int vCount = Vector512<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector512.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector512.LoadUnsafe(ref pSrc, (nuint)i);
                Vector512.Min(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            int vCount = Vector256<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector256.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector256.LoadUnsafe(ref pSrc, (nuint)i);
                Vector256.Min(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            int vCount = Vector128<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector128.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector128.LoadUnsafe(ref pSrc, (nuint)i);
                Vector128.Min(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }

        // Remainder
        for (; i < length; i++)
        {
            dst[i] = Math.Min(dst[i], src[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateMax(Span<float> dst, ReadOnlySpan<float> src)
    {
        int i = 0;
        int length = dst.Length;
        ref float pDst = ref MemoryMarshal.GetReference(dst);
        ref float pSrc = ref MemoryMarshal.GetReference(src);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            int vCount = Vector512<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector512.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector512.LoadUnsafe(ref pSrc, (nuint)i);
                Vector512.Max(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            int vCount = Vector256<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector256.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector256.LoadUnsafe(ref pSrc, (nuint)i);
                Vector256.Max(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            int vCount = Vector128<float>.Count;
            for (; i <= length - vCount; i += vCount)
            {
                var vd = Vector128.LoadUnsafe(ref pDst, (nuint)i);
                var vs = Vector128.LoadUnsafe(ref pSrc, (nuint)i);
                Vector128.Max(vd, vs).StoreUnsafe(ref pDst, (nuint)i);
            }
        }

        // Remainder
        for (; i < length; i++)
        {
            dst[i] = Math.Max(dst[i], src[i]);
        }
    }
}

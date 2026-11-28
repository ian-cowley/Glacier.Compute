// <copyright file="CuDriverTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.DriverTests;

using System;
using System.Runtime.InteropServices;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Drivers;
using Xunit;

[CollectionDefinition("CuDriverSequential", DisableParallelization = true)]
public sealed class CuDriverSequentialCollection
{
}

[Collection("CuDriverSequential")]
public class CuDriverTests
{
    [Fact]
    public void CuDriver_ThrowsPlatformNotSupported_WhenDriverUnavailable()
    {
        using var scope = CuDriver.OverrideExportForTesting(simulateDriverUnavailable: true);

        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.Init(0));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.DeviceGetCount(out _));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.StreamQuery(IntPtr.Zero));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.ModuleLoadData(out _, ".version 8.0"));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.ModuleLoadDataEx(out _, ".version 8.0"));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.MemsetD8(IntPtr.Zero, 0, 64));
        Assert.Throws<PlatformNotSupportedException>(() => CuDriver.MemsetD8Async(IntPtr.Zero, 0, 64, IntPtr.Zero));
        Assert.Throws<PlatformNotSupportedException>(() => CudaDriver.cuInit(0));
    }

    [Theory]
    [InlineData("cuInit")]
    [InlineData("cuDeviceGet")]
    [InlineData("cuCtxSynchronize")]
    [InlineData("cuModuleLoadData")]
    [InlineData("cuModuleLoadDataEx")]
    [InlineData("cuModuleGetFunction")]
    [InlineData("cuLaunchKernel")]
    [InlineData("cuStreamQuery")]
    [InlineData("cuMemAlloc")]
    [InlineData("cuMemsetD8")]
    [InlineData("cuMemsetD8Async")]
    public void CuDriver_ThrowsEntryPointNotFound_WhenExportIsNull(string missingExport)
    {
        using var scope = CuDriver.OverrideExportForTesting(nullExportName: missingExport);

        switch (missingExport)
        {
            case "cuInit":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.Init(0));
                break;
            case "cuDeviceGet":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.DeviceGet(out _, 0));
                break;
            case "cuCtxSynchronize":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.CtxSynchronize());
                break;
            case "cuModuleLoadData":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.ModuleLoadData(out _, ".version 8.0"));
                break;
            case "cuModuleLoadDataEx":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.ModuleLoadDataEx(out _, ".version 8.0"));
                break;
            case "cuModuleGetFunction":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.ModuleGetFunction(out _, IntPtr.Zero, "fn"));
                break;
            case "cuLaunchKernel":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.LaunchKernel(IntPtr.Zero, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero));
                break;
            case "cuStreamQuery":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.StreamQuery(IntPtr.Zero));
                break;
            case "cuMemAlloc":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.MemAlloc(out _, 64));
                break;
            case "cuMemsetD8":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.MemsetD8(IntPtr.Zero, 0, 64));
                break;
            case "cuMemsetD8Async":
                Assert.Throws<EntryPointNotFoundException>(() => CuDriver.MemsetD8Async(IntPtr.Zero, 0, 64, IntPtr.Zero));
                break;
        }
    }

    [Fact]
    public void CuDriver_ArgumentValidationAndCheckHelpers()
    {
        Assert.Throws<ArgumentNullException>(() => CuDriver.DeviceGetName((byte[])null!, 16, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => CuDriver.DeviceGetName(new byte[16], -1, 0));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleLoadData(out _, (byte[])null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleLoadData(out _, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleLoadDataEx(out _, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleGetFunction(out _, IntPtr.Zero, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleGetGlobal(out _, out _, IntPtr.Zero, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.LaunchKernel(IntPtr.Zero, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr[])null!));

        CuDriver.Check(0, "noop");
        var ex = Assert.Throws<InvalidOperationException>(() => CuDriver.Check(200, "test_op"));
        Assert.Contains("test_op", ex.Message);
        Assert.Contains("200", ex.Message);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public unsafe void CuDriver_QueriesDeviceCapabilitiesAndExecutesMemcpyStreamLifecycle_OnNvidiaGpu()
    {
        if (!CuDriver.IsAvailable())
        {
            return;
        }

        CuDriver.Check(CuDriver.DriverGetVersion(out int driverVersion), nameof(CuDriver.DriverGetVersion));
        Assert.True(driverVersion > 0);

        CuDriver.Check(CuDriver.DeviceGetCount(out int count), nameof(CuDriver.DeviceGetCount));
        Assert.True(count > 0);

        CuDriver.Check(CuDriver.DeviceGet(out int dev, 0), nameof(CuDriver.DeviceGet));
        string name = CuDriver.GetDeviceName(dev);
        string cc = CuDriver.GetComputeCapability(dev);
        int smCount = CuDriver.GetMultiprocessorCount(dev);
        int maxThreads = CuDriver.GetMaxThreadsPerBlock(dev);

        Assert.False(string.IsNullOrWhiteSpace(name));
        Assert.StartsWith("sm_", cc, StringComparison.Ordinal);
        Assert.True(smCount > 0);
        Assert.True(maxThreads >= 256);

        CuDriver.Check(CuDriver.DeviceTotalMem(out nuint totalMem, dev), nameof(CuDriver.DeviceTotalMem));
        Assert.True(totalMem > 0);

        IntPtr ctx = IntPtr.Zero;
        IntPtr dBuf = IntPtr.Zero;
        IntPtr stream = IntPtr.Zero;
        IntPtr evStart = IntPtr.Zero;
        IntPtr evEnd = IntPtr.Zero;

        try
        {
            CuDriver.Check(CuDriver.CtxCreate(out ctx, 0, dev), nameof(CuDriver.CtxCreate));
            Assert.NotEqual(IntPtr.Zero, ctx);

            CuDriver.Check(CuDriver.MemGetInfo(out nuint freeBytes, out nuint totalCtxBytes), nameof(CuDriver.MemGetInfo));
            Assert.True(totalCtxBytes > 0 && freeBytes > 0);

            const int elemCount = 64;
            nuint byteCount = (nuint)(elemCount * sizeof(float));
            CuDriver.Check(CuDriver.MemAlloc(out dBuf, byteCount), nameof(CuDriver.MemAlloc));
            CuDriver.Check(CuDriver.MemsetD8(dBuf, 0, byteCount), nameof(CuDriver.MemsetD8));

            float[] src = new float[elemCount];
            float[] dst = new float[elemCount];
            for (int i = 0; i < elemCount; i++)
            {
                src[i] = i * 1.5f;
            }

            CuDriver.Check(CuDriver.StreamCreate(out stream, CuDriver.CU_STREAM_NON_BLOCKING), nameof(CuDriver.StreamCreate));
            CuDriver.Check(CuDriver.EventCreate(out evStart, CuDriver.CU_EVENT_DEFAULT), nameof(CuDriver.EventCreate));
            CuDriver.Check(CuDriver.EventCreate(out evEnd, CuDriver.CU_EVENT_DEFAULT), nameof(CuDriver.EventCreate));

            fixed (float* pSrc = src)
            fixed (float* pDst = dst)
            {
                CuDriver.Check(CuDriver.EventRecord(evStart, stream), nameof(CuDriver.EventRecord));
                CuDriver.Check(CuDriver.MemcpyHtoDAsync(dBuf, (IntPtr)pSrc, byteCount, stream), nameof(CuDriver.MemcpyHtoDAsync));
                CuDriver.Check(CuDriver.MemcpyDtoHAsync((IntPtr)pDst, dBuf, byteCount, stream), nameof(CuDriver.MemcpyDtoHAsync));
                CuDriver.Check(CuDriver.EventRecord(evEnd, stream), nameof(CuDriver.EventRecord));
                CuDriver.Check(CuDriver.StreamSynchronize(stream), nameof(CuDriver.StreamSynchronize));
            }

            CuDriver.Check(CuDriver.EventElapsedTime(out float elapsedMs, evStart, evEnd), nameof(CuDriver.EventElapsedTime));
            Assert.True(elapsedMs >= 0f);
            Assert.Equal(src, dst);

            byte[] fillReadback = new byte[elemCount * sizeof(float)];
            fixed (byte* pFillReadback = fillReadback)
            {
                CuDriver.Check(CuDriver.MemsetD8Async(dBuf, 0x5C, byteCount, stream), nameof(CuDriver.MemsetD8Async));
                CuDriver.Check(CuDriver.MemcpyDtoHAsync((IntPtr)pFillReadback, dBuf, byteCount, stream), nameof(CuDriver.MemcpyDtoHAsync));
                CuDriver.Check(CuDriver.StreamSynchronize(stream), nameof(CuDriver.StreamSynchronize));
            }

            Assert.All(fillReadback, b => Assert.Equal((byte)0x5C, b));
        }
        finally
        {
            if (evEnd != IntPtr.Zero) _ = CuDriver.EventDestroy(evEnd);
            if (evStart != IntPtr.Zero) _ = CuDriver.EventDestroy(evStart);
            if (stream != IntPtr.Zero) _ = CuDriver.StreamDestroy(stream);
            if (dBuf != IntPtr.Zero) _ = CuDriver.MemFree(dBuf);
            if (ctx != IntPtr.Zero) _ = CuDriver.CtxDestroy(ctx);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public unsafe void CudaDriver_JitCompilesAndExecutesPtxEmitterKernel_OnNvidiaGpu()
    {
        if (!CudaDriver.IsAvailable)
        {
            return;
        }

        Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuDeviceGet(out IntPtr devPtr, 0));
        string targetArch = CuDriver.GetComputeCapability((int)devPtr);

        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.GlobalId, 0, 0, 0),    // r0 = gid
            new((int)ComputeOpcode.LoadGlobal, 0, 0, 1),  // f1 = a[r0]
            new((int)ComputeOpcode.LoadGlobal, 1, 0, 2),  // f2 = b[r0]
            new((int)ComputeOpcode.Add, 1, 2, 3),         // f3 = f1 + f2
            new((int)ComputeOpcode.StoreGlobal, 3, 0, 0), // out[r0] = f3
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        string ptx = PtxEmitter.Emit(instructions, "vector_add", targetArch);

        const int n = 137; // non-multiple of block size to exercise param_n boundary guard
        float[] a = new float[n];
        float[] b = new float[n];
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i + 0.25f;
            b[i] = (i * 2) - 0.75f;
        }

        IntPtr ctx = IntPtr.Zero;
        IntPtr mod = IntPtr.Zero;
        IntPtr dA = IntPtr.Zero;
        IntPtr dB = IntPtr.Zero;
        IntPtr dOut = IntPtr.Zero;

        try
        {
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuCtxCreate(out ctx, 0, devPtr));
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuModuleLoadDataEx(out mod, ptx));
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuModuleGetFunction(out IntPtr fn, mod, "vector_add"));
            Assert.NotEqual(IntPtr.Zero, fn);

            nuint bytes = (nuint)(n * sizeof(float));
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemAlloc(out dA, bytes));
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemAlloc(out dB, bytes));
            Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemAlloc(out dOut, bytes));

            fixed (float* pA = a)
            fixed (float* pB = b)
            fixed (float* pOut = result)
            {
                Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemcpyHtoD(dA, pA, bytes));
                Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemcpyHtoD(dB, pB, bytes));

                IntPtr[] kernelParams = { dA, dB, dOut, new IntPtr(n) };
                const uint blockSize = 64;
                uint gridSize = (n + blockSize - 1) / blockSize;

                Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuLaunchKernel(
                    fn,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0,
                    IntPtr.Zero,
                    kernelParams));

                Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuCtxSynchronize());
                Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuMemcpyDtoH(pOut, dOut, bytes));
            }

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(a[i] + b[i], result[i], 4);
            }
        }
        finally
        {
            if (dOut != IntPtr.Zero) _ = CudaDriver.cuMemFree(dOut);
            if (dB != IntPtr.Zero) _ = CudaDriver.cuMemFree(dB);
            if (dA != IntPtr.Zero) _ = CudaDriver.cuMemFree(dA);
            if (mod != IntPtr.Zero) _ = CudaDriver.cuModuleUnload(mod);
            if (ctx != IntPtr.Zero) _ = CudaDriver.cuCtxDestroy(ctx);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public unsafe void CuDriver_JitCompilesAndExecutesSharedMemoryAndShuffleKernel_OnNvidiaGpu()
    {
        if (!CuDriver.IsAvailable())
        {
            return;
        }

        CuDriver.Check(CuDriver.DeviceGet(out int dev, 0), nameof(CuDriver.DeviceGet));
        string targetArch = CuDriver.GetComputeCapability(dev);

        // Exercises LocalId, GlobalId, LoadGlobal, StoreShared, BarrierSync, LoadShared, ShflSyncBfly (vreg 60/61), StoreGlobal
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 60),
            new((int)ComputeOpcode.GlobalId, 0, 0, 61),
            new((int)ComputeOpcode.LoadGlobal, 0, 61, 1),
            new((int)ComputeOpcode.StoreShared, 1, 60, 0),
            new((int)ComputeOpcode.BarrierSync, 0, 0, 0),
            new((int)ComputeOpcode.LoadShared, 0, 60, 2),
            new((int)ComputeOpcode.ShflSyncBfly, 2, 1, 3), // swap adjacent lane pairs
            new((int)ComputeOpcode.StoreGlobal, 3, 61, 0),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        string ptx = PtxEmitter.Emit(instructions, "shfl_shared_pair", targetArch);

        const int n = 32;
        float[] input = new float[n];
        float[] output = new float[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = i + 1.0f;
        }

        IntPtr ctx = IntPtr.Zero;
        IntPtr mod = IntPtr.Zero;
        IntPtr dA = IntPtr.Zero;
        IntPtr dB = IntPtr.Zero;
        IntPtr dOut = IntPtr.Zero;

        try
        {
            CuDriver.Check(CuDriver.CtxCreate(out ctx, 0, dev), nameof(CuDriver.CtxCreate));
            CuDriver.Check(CuDriver.ModuleLoadDataEx(out mod, ptx), nameof(CuDriver.ModuleLoadDataEx));
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fn, mod, "shfl_shared_pair"), nameof(CuDriver.ModuleGetFunction));

            nuint bytes = (nuint)(n * sizeof(float));
            CuDriver.Check(CuDriver.MemAlloc(out dA, bytes), nameof(CuDriver.MemAlloc));
            CuDriver.Check(CuDriver.MemAlloc(out dB, bytes), nameof(CuDriver.MemAlloc));
            CuDriver.Check(CuDriver.MemAlloc(out dOut, bytes), nameof(CuDriver.MemAlloc));

            fixed (float* pIn = input)
            fixed (float* pOut = output)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(dA, (IntPtr)pIn, bytes), nameof(CuDriver.MemcpyHtoD));

                IntPtr[] kernelParams = { dA, dB, dOut, new IntPtr(n) };
                CuDriver.Check(CuDriver.LaunchKernel(
                    fn,
                    1, 1, 1,
                    32, 1, 1,
                    0,
                    IntPtr.Zero,
                    kernelParams), nameof(CuDriver.LaunchKernel));

                CuDriver.Check(CuDriver.CtxSynchronize(), nameof(CuDriver.CtxSynchronize));
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pOut, dOut, bytes), nameof(CuDriver.MemcpyDtoH));
            }

            for (int i = 0; i < n; i++)
            {
                float expected = input[i ^ 1];
                Assert.Equal(expected, output[i], 4);
            }
        }
        finally
        {
            if (dOut != IntPtr.Zero) _ = CuDriver.MemFree(dOut);
            if (dB != IntPtr.Zero) _ = CuDriver.MemFree(dB);
            if (dA != IntPtr.Zero) _ = CuDriver.MemFree(dA);
            if (mod != IntPtr.Zero) _ = CuDriver.ModuleUnload(mod);
            if (ctx != IntPtr.Zero) _ = CuDriver.CtxDestroy(ctx);
        }
    }
}

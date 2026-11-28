// <copyright file="CudaDriver.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Ptx;

using System;
using Glacier.Compute.Drivers;

/// <summary>
/// CUDA Driver API result codes.
/// </summary>
public enum CUresult : int
{
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_UNKNOWN = 999
}

/// <summary>
/// Backwards-compatible PTX-namespace facade over <see cref="CuDriver"/>.
/// Preserves the v1.1.1 <see cref="CUresult"/> API surface while inheriting <see cref="CuDriver"/>'s
/// per-export null checks, fixed <c>CUdevice</c> ABI, and zero-allocation marshaling.
/// </summary>
public static unsafe class CudaDriver
{
    /// <summary>
    /// Gets a value indicating whether the NVIDIA CUDA driver is loaded and initialized with at least one device.
    /// </summary>
    public static bool IsAvailable => CuDriver.IsAvailable();

    /// <summary>Initializes the CUDA driver API.</summary>
    public static CUresult cuInit(uint flags) => (CUresult)CuDriver.Init(flags);

    /// <summary>Returns a handle to a compute device.</summary>
    public static CUresult cuDeviceGet(out IntPtr device, int ordinal)
    {
        int res = CuDriver.DeviceGet(out int dev, ordinal);
        device = (IntPtr)dev;
        return (CUresult)res;
    }

    /// <summary>Creates a CUDA context for the specified device.</summary>
    public static CUresult cuCtxCreate(out IntPtr pctx, uint flags, IntPtr dev)
        => (CUresult)CuDriver.CtxCreate(out pctx, flags, checked((int)dev));

    /// <summary>Destroys a CUDA context.</summary>
    public static CUresult cuCtxDestroy(IntPtr ctx) => (CUresult)CuDriver.CtxDestroy(ctx);

    /// <summary>Blocks until the device has completed all preceding requested tasks.</summary>
    public static CUresult cuCtxSynchronize() => (CUresult)CuDriver.CtxSynchronize();

    /// <summary>Loads a compute module from PTX assembly data.</summary>
    public static CUresult cuModuleLoadDataEx(out IntPtr module, string ptxSource)
        => (CUresult)CuDriver.ModuleLoadDataEx(out module, ptxSource);

    /// <summary>Unloads a compute module.</summary>
    public static CUresult cuModuleUnload(IntPtr module) => (CUresult)CuDriver.ModuleUnload(module);

    /// <summary>Returns a function handle from a module using zero-allocation UTF-8 marshaling.</summary>
    public static CUresult cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name)
        => (CUresult)CuDriver.ModuleGetFunction(out hfunc, hmod, name);

    /// <summary>Launches a compute kernel on the GPU.</summary>
    public static CUresult cuLaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr hStream,
        IntPtr[] kernelParams)
        => (CUresult)CuDriver.LaunchKernel(
            f,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            hStream,
            kernelParams);

    /// <summary>Allocates device VRAM.</summary>
    public static CUresult cuMemAlloc(out IntPtr dptr, nuint bytesize) => (CUresult)CuDriver.MemAlloc(out dptr, bytesize);

    /// <summary>Frees allocated device VRAM.</summary>
    public static CUresult cuMemFree(IntPtr dptr) => (CUresult)CuDriver.MemFree(dptr);

    /// <summary>Copies memory from host to device.</summary>
    public static CUresult cuMemcpyHtoD(IntPtr dstDevice, void* srcHost, nuint byteCount)
        => (CUresult)CuDriver.MemcpyHtoD(dstDevice, srcHost, byteCount);

    /// <summary>Copies memory from device to host.</summary>
    public static CUresult cuMemcpyDtoH(void* dstHost, IntPtr srcDevice, nuint byteCount)
        => (CUresult)CuDriver.MemcpyDtoH(dstHost, srcDevice, byteCount);
}

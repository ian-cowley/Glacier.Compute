// <copyright file="CudaDriver.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Ptx;

using System;
using System.Runtime.InteropServices;
using System.Text;

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
/// Direct pure C# P/Invoke bindings to the NVIDIA CUDA Driver API (nvcuda.dll / libcuda.so).
/// Eliminates runtime CUDA runtime libraries (cudart) and enables zero-latency JIT module loading.
/// </summary>
public static unsafe class CudaDriver
{
    private static readonly IntPtr s_driverHandle;
    private static readonly bool s_isAvailable;

    // Delegates for driver entry points
    private delegate CUresult CuInitDelegate(uint flags);
    private delegate CUresult CuDeviceGetDelegate(out IntPtr device, int ordinal);
    private delegate CUresult CuCtxCreateDelegate(out IntPtr pctx, uint flags, IntPtr dev);
    private delegate CUresult CuCtxDestroyDelegate(IntPtr ctx);
    private delegate CUresult CuCtxSynchronizeDelegate();
    private delegate CUresult CuModuleLoadDataExDelegate(out IntPtr module, byte* image, uint numOptions, IntPtr options, IntPtr optionValues);
    private delegate CUresult CuModuleLoadDataDelegate(out IntPtr module, byte* image);
    private delegate CUresult CuModuleUnloadDelegate(IntPtr module);
    private delegate CUresult CuModuleGetFunctionDelegate(out IntPtr hfunc, IntPtr hmod, [MarshalAs(UnmanagedType.LPStr)] string name);
    private delegate CUresult CuLaunchKernelDelegate(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr hStream,
        void** kernelParams,
        void** extra);
    private delegate CUresult CuMemAllocDelegate(out IntPtr dptr, nuint bytesize);
    private delegate CUresult CuMemFreeDelegate(IntPtr dptr);
    private delegate CUresult CuMemcpyHtoDDelegate(IntPtr dstDevice, void* srcHost, nuint ByteCount);
    private delegate CUresult CuMemcpyDtoHDelegate(void* dstHost, IntPtr srcDevice, nuint ByteCount);

    private static readonly CuInitDelegate? s_cuInit;
    private static readonly CuDeviceGetDelegate? s_cuDeviceGet;
    private static readonly CuCtxCreateDelegate? s_cuCtxCreate;
    private static readonly CuCtxDestroyDelegate? s_cuCtxDestroy;
    private static readonly CuCtxSynchronizeDelegate? s_cuCtxSynchronize;
    private static readonly CuModuleLoadDataExDelegate? s_cuModuleLoadDataEx;
    private static readonly CuModuleLoadDataDelegate? s_cuModuleLoadData;
    private static readonly CuModuleUnloadDelegate? s_cuModuleUnload;
    private static readonly CuModuleGetFunctionDelegate? s_cuModuleGetFunction;
    private static readonly CuLaunchKernelDelegate? s_cuLaunchKernel;
    private static readonly CuMemAllocDelegate? s_cuMemAlloc;
    private static readonly CuMemFreeDelegate? s_cuMemFree;
    private static readonly CuMemcpyHtoDDelegate? s_cuMemcpyHtoD;
    private static readonly CuMemcpyDtoHDelegate? s_cuMemcpyDtoH;

    static CudaDriver()
    {
        string libraryName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so";
        if (NativeLibrary.TryLoad(libraryName, typeof(CudaDriver).Assembly, null, out s_driverHandle) ||
            NativeLibrary.TryLoad("libcuda.so.1", typeof(CudaDriver).Assembly, null, out s_driverHandle))
        {
            s_cuInit = GetProc<CuInitDelegate>("cuInit");
            s_cuDeviceGet = GetProc<CuDeviceGetDelegate>("cuDeviceGet");
            s_cuCtxCreate = GetProc<CuCtxCreateDelegate>("cuCtxCreate_v2") ?? GetProc<CuCtxCreateDelegate>("cuCtxCreate");
            s_cuCtxDestroy = GetProc<CuCtxDestroyDelegate>("cuCtxDestroy_v2") ?? GetProc<CuCtxDestroyDelegate>("cuCtxDestroy");
            s_cuCtxSynchronize = GetProc<CuCtxSynchronizeDelegate>("cuCtxSynchronize");
            s_cuModuleLoadDataEx = GetProc<CuModuleLoadDataExDelegate>("cuModuleLoadDataEx");
            s_cuModuleLoadData = GetProc<CuModuleLoadDataDelegate>("cuModuleLoadData");
            s_cuModuleUnload = GetProc<CuModuleUnloadDelegate>("cuModuleUnload");
            s_cuModuleGetFunction = GetProc<CuModuleGetFunctionDelegate>("cuModuleGetFunction");
            s_cuLaunchKernel = GetProc<CuLaunchKernelDelegate>("cuLaunchKernel");
            s_cuMemAlloc = GetProc<CuMemAllocDelegate>("cuMemAlloc_v2") ?? GetProc<CuMemAllocDelegate>("cuMemAlloc");
            s_cuMemFree = GetProc<CuMemFreeDelegate>("cuMemFree_v2") ?? GetProc<CuMemFreeDelegate>("cuMemFree");
            s_cuMemcpyHtoD = GetProc<CuMemcpyHtoDDelegate>("cuMemcpyHtoD_v2") ?? GetProc<CuMemcpyHtoDDelegate>("cuMemcpyHtoD");
            s_cuMemcpyDtoH = GetProc<CuMemcpyDtoHDelegate>("cuMemcpyDtoH_v2") ?? GetProc<CuMemcpyDtoHDelegate>("cuMemcpyDtoH");

            if (s_cuInit != null)
            {
                var res = s_cuInit(0);
                s_isAvailable = (res == CUresult.CUDA_SUCCESS);
            }
        }
    }

    private static T? GetProc<T>(string name) where T : Delegate
    {
        if (s_driverHandle != IntPtr.Zero && NativeLibrary.TryGetExport(s_driverHandle, name, out IntPtr address))
        {
            return Marshal.GetDelegateForFunctionPointer<T>(address);
        }
        return null;
    }

    /// <summary>
    /// Gets a value indicating whether the NVIDIA CUDA driver is loaded and initialized.
    /// </summary>
    public static bool IsAvailable => s_isAvailable;

    /// <summary>Initializes the CUDA driver API.</summary>
    public static CUresult cuInit(uint flags)
    {
        EnsureAvailable();
        return s_cuInit!(flags);
    }

    /// <summary>Returns a handle to a compute device.</summary>
    public static CUresult cuDeviceGet(out IntPtr device, int ordinal)
    {
        EnsureAvailable();
        return s_cuDeviceGet!(out device, ordinal);
    }

    /// <summary>Creates a CUDA context for the specified device.</summary>
    public static CUresult cuCtxCreate(out IntPtr pctx, uint flags, IntPtr dev)
    {
        EnsureAvailable();
        return s_cuCtxCreate!(out pctx, flags, dev);
    }

    /// <summary>Destroys a CUDA context.</summary>
    public static CUresult cuCtxDestroy(IntPtr ctx)
    {
        EnsureAvailable();
        return s_cuCtxDestroy!(ctx);
    }

    /// <summary>Blocks until the device has completed all preceding requested tasks.</summary>
    public static CUresult cuCtxSynchronize()
    {
        EnsureAvailable();
        return s_cuCtxSynchronize!();
    }

    /// <summary>Loads a compute module from PTX assembly data.</summary>
    public static CUresult cuModuleLoadDataEx(out IntPtr module, string ptxSource)
    {
        EnsureAvailable();
        byte[] bytes = Encoding.UTF8.GetBytes(ptxSource + "\0");
        fixed (byte* pBytes = bytes)
        {
            if (s_cuModuleLoadDataEx != null)
            {
                return s_cuModuleLoadDataEx(out module, pBytes, 0, IntPtr.Zero, IntPtr.Zero);
            }
            if (s_cuModuleLoadData != null)
            {
                return s_cuModuleLoadData(out module, pBytes);
            }
        }
        throw new NotSupportedException("cuModuleLoadData not found in CUDA driver.");
    }

    /// <summary>Unloads a compute module.</summary>
    public static CUresult cuModuleUnload(IntPtr module)
    {
        EnsureAvailable();
        return s_cuModuleUnload!(module);
    }

    /// <summary>Returns a function handle from a module.</summary>
    public static CUresult cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name)
    {
        EnsureAvailable();
        return s_cuModuleGetFunction!(out hfunc, hmod, name);
    }

    /// <summary>Launches a compute kernel on the GPU.</summary>
    public static CUresult cuLaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr hStream,
        IntPtr[] kernelParams)
    {
        EnsureAvailable();
        void*[] paramPointers = new void*[kernelParams.Length];
        fixed (IntPtr* pParams = kernelParams)
        {
            for (int i = 0; i < kernelParams.Length; i++)
            {
                paramPointers[i] = &pParams[i];
            }
            fixed (void** ppParams = paramPointers)
            {
                return s_cuLaunchKernel!(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, ppParams, null);
            }
        }
    }

    /// <summary>Allocates device VRAM.</summary>
    public static CUresult cuMemAlloc(out IntPtr dptr, nuint bytesize)
    {
        EnsureAvailable();
        return s_cuMemAlloc!(out dptr, bytesize);
    }

    /// <summary>Frees allocated device VRAM.</summary>
    public static CUresult cuMemFree(IntPtr dptr)
    {
        EnsureAvailable();
        return s_cuMemFree!(dptr);
    }

    /// <summary>Copies memory from host to device.</summary>
    public static CUresult cuMemcpyHtoD(IntPtr dstDevice, void* srcHost, nuint byteCount)
    {
        EnsureAvailable();
        return s_cuMemcpyHtoD!(dstDevice, srcHost, byteCount);
    }

    /// <summary>Copies memory from device to host.</summary>
    public static CUresult cuMemcpyDtoH(void* dstHost, IntPtr srcDevice, nuint byteCount)
    {
        EnsureAvailable();
        return s_cuMemcpyDtoH!(dstHost, srcDevice, byteCount);
    }

    private static void EnsureAvailable()
    {
        if (!s_isAvailable)
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA Driver (nvcuda.dll / libcuda.so) is not available or initialized on this system.");
        }
    }
}

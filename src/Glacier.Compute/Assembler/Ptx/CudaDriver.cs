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
/// Direct Native AOT-safe function-pointer bindings to the NVIDIA CUDA Driver API (<c>nvcuda.dll</c> / <c>libcuda.so</c>).
/// Eliminates runtime CUDA libraries (<c>cudart</c>) and dynamic marshaling stubs for zero-latency JIT module loading.
/// </summary>
public static unsafe class CudaDriver
{
    private static readonly IntPtr s_driverHandle;
    private static readonly bool s_isAvailable;

    private static readonly delegate* unmanaged<uint, CUresult> s_cuInit;
    private static readonly delegate* unmanaged<IntPtr*, int, CUresult> s_cuDeviceGet;
    private static readonly delegate* unmanaged<IntPtr*, uint, IntPtr, CUresult> s_cuCtxCreate;
    private static readonly delegate* unmanaged<IntPtr, CUresult> s_cuCtxDestroy;
    private static readonly delegate* unmanaged<CUresult> s_cuCtxSynchronize;
    private static readonly delegate* unmanaged<IntPtr*, byte*, uint, IntPtr, IntPtr, CUresult> s_cuModuleLoadDataEx;
    private static readonly delegate* unmanaged<IntPtr*, byte*, CUresult> s_cuModuleLoadData;
    private static readonly delegate* unmanaged<IntPtr, CUresult> s_cuModuleUnload;
    private static readonly delegate* unmanaged<IntPtr*, IntPtr, byte*, CUresult> s_cuModuleGetFunction;
    private static readonly delegate* unmanaged<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, CUresult> s_cuLaunchKernel;
    private static readonly delegate* unmanaged<IntPtr*, nuint, CUresult> s_cuMemAlloc;
    private static readonly delegate* unmanaged<IntPtr, CUresult> s_cuMemFree;
    private static readonly delegate* unmanaged<IntPtr, void*, nuint, CUresult> s_cuMemcpyHtoD;
    private static readonly delegate* unmanaged<void*, IntPtr, nuint, CUresult> s_cuMemcpyDtoH;

    static CudaDriver()
    {
        string libraryName = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so";
        if (NativeLibrary.TryLoad(libraryName, typeof(CudaDriver).Assembly, null, out s_driverHandle) ||
            NativeLibrary.TryLoad("libcuda.so.1", typeof(CudaDriver).Assembly, null, out s_driverHandle))
        {
            s_cuInit = (delegate* unmanaged<uint, CUresult>)GetProcAddress("cuInit");
            s_cuDeviceGet = (delegate* unmanaged<IntPtr*, int, CUresult>)GetProcAddress("cuDeviceGet");
            s_cuCtxCreate = (delegate* unmanaged<IntPtr*, uint, IntPtr, CUresult>)GetProcAddress("cuCtxCreate_v2", "cuCtxCreate");
            s_cuCtxDestroy = (delegate* unmanaged<IntPtr, CUresult>)GetProcAddress("cuCtxDestroy_v2", "cuCtxDestroy");
            s_cuCtxSynchronize = (delegate* unmanaged<CUresult>)GetProcAddress("cuCtxSynchronize");
            s_cuModuleLoadDataEx = (delegate* unmanaged<IntPtr*, byte*, uint, IntPtr, IntPtr, CUresult>)GetProcAddress("cuModuleLoadDataEx");
            s_cuModuleLoadData = (delegate* unmanaged<IntPtr*, byte*, CUresult>)GetProcAddress("cuModuleLoadData");
            s_cuModuleUnload = (delegate* unmanaged<IntPtr, CUresult>)GetProcAddress("cuModuleUnload");
            s_cuModuleGetFunction = (delegate* unmanaged<IntPtr*, IntPtr, byte*, CUresult>)GetProcAddress("cuModuleGetFunction");
            s_cuLaunchKernel = (delegate* unmanaged<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, CUresult>)GetProcAddress("cuLaunchKernel");
            s_cuMemAlloc = (delegate* unmanaged<IntPtr*, nuint, CUresult>)GetProcAddress("cuMemAlloc_v2", "cuMemAlloc");
            s_cuMemFree = (delegate* unmanaged<IntPtr, CUresult>)GetProcAddress("cuMemFree_v2", "cuMemFree");
            s_cuMemcpyHtoD = (delegate* unmanaged<IntPtr, void*, nuint, CUresult>)GetProcAddress("cuMemcpyHtoD_v2", "cuMemcpyHtoD");
            s_cuMemcpyDtoH = (delegate* unmanaged<void*, IntPtr, nuint, CUresult>)GetProcAddress("cuMemcpyDtoH_v2", "cuMemcpyDtoH");

            if (s_cuInit != null)
            {
                var res = s_cuInit(0);
                s_isAvailable = (res == CUresult.CUDA_SUCCESS);
            }
        }
    }

    private static IntPtr GetProcAddress(string primaryName, string? fallbackName = null)
    {
        if (s_driverHandle != IntPtr.Zero)
        {
            if (NativeLibrary.TryGetExport(s_driverHandle, primaryName, out IntPtr address))
            {
                return address;
            }

            if (fallbackName is not null && NativeLibrary.TryGetExport(s_driverHandle, fallbackName, out address))
            {
                return address;
            }
        }

        return IntPtr.Zero;
    }

    /// <summary>
    /// Gets a value indicating whether the NVIDIA CUDA driver is loaded and initialized.
    /// </summary>
    public static bool IsAvailable => s_isAvailable;

    /// <summary>Initializes the CUDA driver API.</summary>
    public static CUresult cuInit(uint flags)
    {
        EnsureAvailable();
        return s_cuInit(flags);
    }

    /// <summary>Returns a handle to a compute device.</summary>
    public static CUresult cuDeviceGet(out IntPtr device, int ordinal)
    {
        EnsureAvailable();
        IntPtr dev = IntPtr.Zero;
        CUresult res = s_cuDeviceGet(&dev, ordinal);
        device = dev;
        return res;
    }

    /// <summary>Creates a CUDA context for the specified device.</summary>
    public static CUresult cuCtxCreate(out IntPtr pctx, uint flags, IntPtr dev)
    {
        EnsureAvailable();
        IntPtr ctx = IntPtr.Zero;
        CUresult res = s_cuCtxCreate(&ctx, flags, dev);
        pctx = ctx;
        return res;
    }

    /// <summary>Destroys a CUDA context.</summary>
    public static CUresult cuCtxDestroy(IntPtr ctx)
    {
        EnsureAvailable();
        return s_cuCtxDestroy(ctx);
    }

    /// <summary>Blocks until the device has completed all preceding requested tasks.</summary>
    public static CUresult cuCtxSynchronize()
    {
        EnsureAvailable();
        return s_cuCtxSynchronize();
    }

    /// <summary>Loads a compute module from PTX assembly data.</summary>
    public static CUresult cuModuleLoadDataEx(out IntPtr module, string ptxSource)
    {
        ArgumentNullException.ThrowIfNull(ptxSource);
        EnsureAvailable();

        int utf8ByteCount = Encoding.UTF8.GetByteCount(ptxSource);
        byte[] bytes = new byte[utf8ByteCount + 1];
        Encoding.UTF8.GetBytes(ptxSource, bytes.AsSpan(0, utf8ByteCount));

        IntPtr mod = IntPtr.Zero;
        fixed (byte* pBytes = bytes)
        {
            if (s_cuModuleLoadDataEx != null)
            {
                CUresult res = s_cuModuleLoadDataEx(&mod, pBytes, 0, IntPtr.Zero, IntPtr.Zero);
                module = mod;
                return res;
            }

            if (s_cuModuleLoadData != null)
            {
                CUresult res = s_cuModuleLoadData(&mod, pBytes);
                module = mod;
                return res;
            }
        }

        throw new NotSupportedException("cuModuleLoadData not found in CUDA driver.");
    }

    /// <summary>Unloads a compute module.</summary>
    public static CUresult cuModuleUnload(IntPtr module)
    {
        EnsureAvailable();
        return s_cuModuleUnload(module);
    }

    /// <summary>Returns a function handle from a module using blittable UTF-8 marshaling.</summary>
    public static CUresult cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        EnsureAvailable();

        int utf8Len = Encoding.UTF8.GetByteCount(name);
        byte[]? rented = null;
        Span<byte> utf8Buffer = utf8Len < 256
            ? stackalloc byte[utf8Len + 1]
            : (rented = new byte[utf8Len + 1]);

        Encoding.UTF8.GetBytes(name, utf8Buffer[..utf8Len]);
        utf8Buffer[utf8Len] = 0;

        IntPtr fn = IntPtr.Zero;
        fixed (byte* pName = utf8Buffer)
        {
            CUresult res = s_cuModuleGetFunction(&fn, hmod, pName);
            hfunc = fn;
            _ = rented;
            return res;
        }
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
        ArgumentNullException.ThrowIfNull(kernelParams);
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
                return s_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, ppParams, null);
            }
        }
    }

    /// <summary>Allocates device VRAM.</summary>
    public static CUresult cuMemAlloc(out IntPtr dptr, nuint bytesize)
    {
        EnsureAvailable();
        IntPtr ptr = IntPtr.Zero;
        CUresult res = s_cuMemAlloc(&ptr, bytesize);
        dptr = ptr;
        return res;
    }

    /// <summary>Frees allocated device VRAM.</summary>
    public static CUresult cuMemFree(IntPtr dptr)
    {
        EnsureAvailable();
        return s_cuMemFree(dptr);
    }

    /// <summary>Copies memory from host to device.</summary>
    public static CUresult cuMemcpyHtoD(IntPtr dstDevice, void* srcHost, nuint byteCount)
    {
        EnsureAvailable();
        return s_cuMemcpyHtoD(dstDevice, srcHost, byteCount);
    }

    /// <summary>Copies memory from device to host.</summary>
    public static CUresult cuMemcpyDtoH(void* dstHost, IntPtr srcDevice, nuint byteCount)
    {
        EnsureAvailable();
        return s_cuMemcpyDtoH(dstHost, srcDevice, byteCount);
    }

    private static void EnsureAvailable()
    {
        if (!s_isAvailable)
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA Driver (nvcuda.dll / libcuda.so) is not available or initialized on this system.");
        }
    }
}

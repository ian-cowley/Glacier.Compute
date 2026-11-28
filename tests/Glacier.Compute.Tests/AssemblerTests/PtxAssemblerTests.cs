// <copyright file="PtxAssemblerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using System.Text.RegularExpressions;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Drivers;
using Xunit;

public class PtxAssemblerTests
{
    [Fact]
    public void PtxEmitter_EmitsValidPtx80Header()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.LocalId, 0, 0, 1),
            new((int)ComputeOpcode.Add, 2, 3, 4),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        };

        string ptx = PtxEmitter.Emit(instructions, "test_kernel", "sm_90");

        Assert.Contains(".version 8.0", ptx);
        Assert.Contains(".target sm_90", ptx);
        Assert.Contains(".address_size 64", ptx);
        Assert.Contains(".visible .entry test_kernel", ptx);
        Assert.Contains("ret;", ptx);
    }

    [Fact]
    public void PtxEmitter_TranslatesArithmeticOpcodes_WithExactRegisterCounts()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.Sub, 4, 5, 6),
            new((int)ComputeOpcode.Mul, 7, 8, 9),
            new((int)ComputeOpcode.Div, 10, 11, 12),
            new((int)ComputeOpcode.Fma, 13, 14, 15),
            new((int)ComputeOpcode.Neg, 16, 0, 17),
            new((int)ComputeOpcode.Abs, 18, 0, 19),
            new((int)ComputeOpcode.Min, 20, 21, 22),
            new((int)ComputeOpcode.Max, 23, 24, 25),
            new((int)ComputeOpcode.Sqrt, 26, 0, 27),
            new((int)ComputeOpcode.Rsqrt, 28, 0, 29),
            new((int)ComputeOpcode.Exp, 30, 0, 31),
            new((int)ComputeOpcode.Log, 32, 0, 33),
            new((int)ComputeOpcode.Tanh, 34, 0, 35)
        };

        string ptx = PtxEmitter.Emit(instructions);

        Assert.Contains("add.f32 %f", ptx);
        Assert.Contains("sub.f32 %f", ptx);
        Assert.Contains("mul.f32 %f", ptx);
        Assert.Contains("div.rn.f32 %f", ptx);
        Assert.Contains("fma.rn.f32 %f", ptx);
        Assert.Contains("neg.f32 %f", ptx);
        Assert.Contains("abs.f32 %f", ptx);
        Assert.Contains("min.f32 %f", ptx);
        Assert.Contains("max.f32 %f", ptx);
        Assert.Contains("sqrt.rn.f32 %f", ptx);
        Assert.Contains("rsqrt.approx.f32 %f", ptx);
        Assert.Contains(", 0f3FB8AA3B;", ptx);
        Assert.Contains("ex2.approx.f32 %f", ptx);
        Assert.Contains("lg2.approx.f32 %f", ptx);
        Assert.Contains(", 0f3F317218;", ptx);
        Assert.Contains("tanh.approx.f32 %f", ptx);

        // Exact register declarations: 1 pred (%p0), 5 ABI %r, 3 ABI %rd, 35 live %f (vregs 1..35), no unused smem
        Assert.Contains(".reg .pred %p<1>;", ptx);
        Assert.Contains(".reg .b32  %r<5>;", ptx);
        Assert.Contains(".reg .b64  %rd<3>;", ptx);
        Assert.Contains(".reg .f32  %f<35>;", ptx);
        Assert.DoesNotContain("smem_workspace", ptx);
    }

    [Fact]
    public void PtxEmitter_TypedRegisterAllocation_PreventsAbiCollisionAndClobber()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.ShflSyncBfly, 60, 16, 61),
            new((int)ComputeOpcode.BarrierSync, 0, 0, 0),
            new((int)ComputeOpcode.LocalId, 0, 0, 1),   // virtual r1 must NOT clobber ABI param_n (%r0) or gid (%r4)
            new((int)ComputeOpcode.GlobalId, 0, 0, 4),  // virtual r4 must NOT clobber ABI gid (%r4)
            new((int)ComputeOpcode.LoadShared, 0, 4, 8),
            new((int)ComputeOpcode.StoreShared, 8, 1, 0),
            new((int)ComputeOpcode.AtomicAdd, 8, 4, 13),
            new((int)ComputeOpcode.LoadGlobal, 99, 4, 14), // out-of-range buffer index clamps to param_output (%rd2)
            new((int)ComputeOpcode.LoadConstant, BitConverter.SingleToInt32Bits(2.5f), 0, 15),
            new((int)ComputeOpcode.Add, 1, 15, 16), // uses LocalId vreg 1 as float -> cvt.rn.f32.u32 required
            new((int)ComputeOpcode.StoreGlobal, 16, 15, 0) // uses LoadConstant vreg 15 as uint index
        };

        string ptx = PtxEmitter.Emit(instructions);

        // ABI registers %r0..%r4 must NEVER be overwritten in the kernel instruction stream
        string instructionStream = ptx[ptx.IndexOf("// Kernel Instruction Stream", StringComparison.Ordinal)..];
        Assert.DoesNotContain("mov.u32 %r0,", instructionStream);
        Assert.DoesNotContain("mov.u32 %r1,", instructionStream);
        Assert.DoesNotContain("mov.u32 %r2,", instructionStream);
        Assert.DoesNotContain("mov.u32 %r3,", instructionStream);
        Assert.DoesNotContain("mov.u32 %r4,", instructionStream);

        // Virtual registers 60/61 must NOT collide with ShflSyncBfly scratch registers (%r5/%r6)
        Assert.Contains("shfl.sync.bfly.b32 %r6, %r5, 16, 0x1f, 0xffffffff;", ptx);
        Assert.Contains("bar.sync 0;", ptx);
        Assert.Contains("ld.shared.f32", ptx);
        Assert.Contains("st.shared.f32", ptx);
        Assert.Contains("atom.global.add.f32", ptx);
        Assert.Contains("cvt.rn.f32.u32", ptx);
        Assert.Contains("0f40200000;", ptx);
        Assert.Contains("0x40200000;", ptx);

        // Clamped LoadGlobal(99) selects param_output (%rd2) rather than defaulting to %rd0
        Assert.Matches(new Regex(@"add\.u64 %rd\d+, %rd2, %rd\d+;"), ptx);
        Assert.Contains("smem_workspace[4096];", ptx);
    }

    [Fact]
    public void PtxEmitter_ThrowsOnUnknownOpcode_AndInvalidMetadataArguments()
    {
        Assert.Throws<NotSupportedException>(() =>
            PtxEmitter.Emit(new Instruction[] { new(999, 0, 0, 0) }));

        Assert.Throws<NotSupportedException>(() =>
            PtxEmitter.Emit(new Instruction[] { new(-1, 0, 0, 0) }));

        // Invalid opcode even after Return must throw during pre-scan
        Assert.Throws<NotSupportedException>(() =>
            PtxEmitter.Emit(new Instruction[]
            {
                new((int)ComputeOpcode.Return, 0, 0, 0),
                new(777, 0, 0, 0)
            }));

        Assert.Throws<ArgumentException>(() =>
            PtxEmitter.Emit(ReadOnlySpan<Instruction>.Empty, " ", "sm_89"));

        Assert.Throws<ArgumentException>(() =>
            PtxEmitter.Emit(ReadOnlySpan<Instruction>.Empty, "k", " "));
    }

    [Fact]
    public void PtxKernelBuilder_BuildsCustomKernel()
    {
        var builder = new PtxKernelBuilder("gemm_tile")
            .SetTargetArch("sm_89")
            .AddParameter(".u64", "matrix_a")
            .AddParameter(".u64", "matrix_b")
            .AddParameter(".u64", "matrix_c")
            .AddParameter(".u32", "m")
            .AddParameter(".u32", "n")
            .AddParameter(".u32", "k")
            .SetRegisterCounts(b32: 64, b64: 32, f32: 64)
            .SetSharedMemoryBytes(8192)
            .EmitFma("%f10", "%f1", "%f2", "%f3")
            .EmitShuffleBfly("%f4", "%f10", 8)
            .EmitBarrier();

        string ptx = builder.Build();

        Assert.Contains(".version 8.0", ptx);
        Assert.Contains(".target sm_89", ptx);
        Assert.Contains(".visible .entry gemm_tile(", ptx);
        Assert.Contains(".param .u64 matrix_a", ptx);
        Assert.Contains(".shared .align 4 .b8 smem[8192];", ptx);
        Assert.Contains("fma.rn.f32 %f10, %f1, %f2, %f3;", ptx);
        Assert.Contains("shfl.sync.bfly.b32 %f4, %f10, 8, 0x1f, 0xffffffff;", ptx);
        Assert.Contains("bar.sync 0;", ptx);
    }

    [Fact]
    public void CudaDriver_AvailabilityAndGuardContracts()
    {
        Assert.Equal(CuDriver.IsAvailable(), CudaDriver.IsAvailable);

        Assert.Throws<ArgumentNullException>(() => CudaDriver.cuModuleLoadDataEx(out _, null!));
        Assert.Throws<ArgumentNullException>(() => CudaDriver.cuModuleGetFunction(out _, IntPtr.Zero, null!));
        Assert.Throws<ArgumentNullException>(() => CudaDriver.cuLaunchKernel(IntPtr.Zero, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, null!));

        if (!CudaDriver.IsAvailable)
        {
            Assert.Throws<PlatformNotSupportedException>(() => CudaDriver.cuInit(0));
            return;
        }

        Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuInit(0));
        Assert.Equal(CUresult.CUDA_SUCCESS, CudaDriver.cuDeviceGet(out IntPtr dev0, 0));
        Assert.True((int)dev0 >= 0);
    }
}

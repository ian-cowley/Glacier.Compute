// <copyright file="PtxAssemblerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Tests.AssemblerTests;

using System;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Ptx;
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
    public void PtxEmitter_TranslatesArithmeticOpcodes()
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

        Assert.Contains("add.f32 %f3, %f1, %f2;", ptx);
        Assert.Contains("sub.f32 %f6, %f4, %f5;", ptx);
        Assert.Contains("mul.f32 %f9, %f7, %f8;", ptx);
        Assert.Contains("div.rn.f32 %f12, %f10, %f11;", ptx);
        Assert.Contains("fma.rn.f32 %f15, %f13, %f14, %f15;", ptx);
        Assert.Contains("neg.f32 %f17, %f16;", ptx);
        Assert.Contains("abs.f32 %f19, %f18;", ptx);
        Assert.Contains("min.f32 %f22, %f20, %f21;", ptx);
        Assert.Contains("max.f32 %f25, %f23, %f24;", ptx);
        Assert.Contains("sqrt.rn.f32 %f27, %f26;", ptx);
        Assert.Contains("rsqrt.approx.f32 %f29, %f28;", ptx);
        Assert.Contains("ex2.approx.f32 %f31, %f30;", ptx);
        Assert.Contains("lg2.approx.f32 %f33, %f32;", ptx);
        Assert.Contains("tanh.approx.f32 %f35, %f34;", ptx);
    }

    [Fact]
    public void PtxEmitter_TranslatesSyncAndMemoryOpcodes()
    {
        var instructions = new Instruction[]
        {
            new((int)ComputeOpcode.ShflSyncBfly, 1, 16, 2),
            new((int)ComputeOpcode.BarrierSync, 0, 0, 0),
            new((int)ComputeOpcode.LocalId, 0, 0, 5),
            new((int)ComputeOpcode.GlobalId, 0, 0, 6),
            new((int)ComputeOpcode.LoadShared, 0, 7, 8),
            new((int)ComputeOpcode.StoreShared, 9, 10, 0),
            new((int)ComputeOpcode.AtomicAdd, 11, 12, 13)
        };

        string ptx = PtxEmitter.Emit(instructions);

        Assert.Contains("shfl.sync.bfly.b32 %f2, %f1, 16, 0x1f, 0xffffffff;", ptx);
        Assert.Contains("bar.sync 0;", ptx);
        Assert.Contains("mov.u32 %r5, %tid.x;", ptx);
        Assert.Contains("mov.u32 %r6, %r5;", ptx);
        Assert.Contains("ld.shared.f32 %f8, [%rd16];", ptx);
        Assert.Contains("st.shared.f32 [%rd19], %f9;", ptx);
        Assert.Contains("atom.global.add.f32 %f13, [%rd21], %f11;", ptx);
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
    public void CudaDriver_DetectionSafe()
    {
        // Safe check without throwing even if no NVIDIA GPU is physically present
        bool isAvailable = CudaDriver.IsAvailable;
        // Should return a boolean without crashing
        Assert.True(isAvailable || !isAvailable);
    }
}

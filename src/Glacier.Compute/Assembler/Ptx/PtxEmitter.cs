// <copyright file="PtxEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Ptx;

using System;
using System.Globalization;
using System.Text;

/// <summary>
/// High-performance NVIDIA PTX 8.0 JIT assembly generator.
/// Generates compliant PTX 8.0 text for sm_80, sm_89, and sm_90 architectures with typed virtual register allocation.
/// </summary>
public static class PtxEmitter
{
    /// <summary>
    /// Default target architecture for generated PTX.
    /// </summary>
    public const string DefaultTargetArch = "sm_90";

    /// <summary>
    /// Emits PTX 8.0 source text for a standard compute kernel from intermediate instructions.
    /// </summary>
    /// <param name="instructions">The sequence of intermediate instructions.</param>
    /// <param name="kernelName">The name of the entry kernel.</param>
    /// <param name="targetArch">The target compute architecture (e.g. sm_90, sm_89, sm_80).</param>
    /// <returns>A valid PTX 8.0 assembly string.</returns>
    public static string Emit(ReadOnlySpan<Instruction> instructions, string kernelName = "glacier_compute_kernel", string targetArch = DefaultTargetArch)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelName);
        ArgumentException.ThrowIfNullOrWhiteSpace(targetArch);

        // Pre-scan all instructions (including after Return) to reject unknown opcodes and collect live virtual registers.
        ulong usedFloatRegs = 0;
        ulong usedUintRegs = 0;
        bool usesShared = false;
        bool usesShuffle = false;
        bool usesGlobalMem = false;
        bool scanTerminated = false;

        for (int i = 0; i < instructions.Length; i++)
        {
            var inst = instructions[i];
            if (inst.Opcode < (int)ComputeOpcode.Nop || inst.Opcode > (int)ComputeOpcode.Return)
            {
                throw new NotSupportedException($"Unsupported ComputeOpcode '{inst.Opcode}' at instruction index {i} for PTX 8.0 emission.");
            }

            if (scanTerminated)
            {
                continue;
            }

            var opcode = (ComputeOpcode)inst.Opcode;
            int a = Math.Clamp(inst.OperandA, 0, 63);
            int b = Math.Clamp(inst.OperandB, 0, 63);
            int d = Math.Clamp(inst.Destination, 0, 63);

            switch (opcode)
            {
                case ComputeOpcode.Nop:
                case ComputeOpcode.BarrierSync:
                    break;

                case ComputeOpcode.Add:
                case ComputeOpcode.Sub:
                case ComputeOpcode.Mul:
                case ComputeOpcode.Div:
                case ComputeOpcode.Fma:
                case ComputeOpcode.Min:
                case ComputeOpcode.Max:
                    usedFloatRegs |= (1UL << a) | (1UL << b) | (1UL << d);
                    break;

                case ComputeOpcode.Neg:
                case ComputeOpcode.Abs:
                case ComputeOpcode.Sqrt:
                case ComputeOpcode.Rsqrt:
                case ComputeOpcode.Exp:
                case ComputeOpcode.Log:
                case ComputeOpcode.Tanh:
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    break;

                case ComputeOpcode.LoadGlobal:
                    usesGlobalMem = true;
                    usedUintRegs |= (1UL << b);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.StoreGlobal:
                    usesGlobalMem = true;
                    usedFloatRegs |= (1UL << a);
                    usedUintRegs |= (1UL << b);
                    break;

                case ComputeOpcode.LoadShared:
                    usesShared = true;
                    usedUintRegs |= (1UL << b);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.StoreShared:
                    usesShared = true;
                    usedFloatRegs |= (1UL << a);
                    usedUintRegs |= (1UL << b);
                    break;

                case ComputeOpcode.LoadConstant:
                    usedFloatRegs |= (1UL << d);
                    usedUintRegs |= (1UL << d);
                    break;

                case ComputeOpcode.ShflSyncBfly:
                    usesShuffle = true;
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    break;

                case ComputeOpcode.LocalId:
                case ComputeOpcode.GlobalId:
                    usedUintRegs |= (1UL << d);
                    usedFloatRegs |= (1UL << d);
                    break;

                case ComputeOpcode.AtomicAdd:
                    usesGlobalMem = true;
                    usedFloatRegs |= (1UL << a) | (1UL << d);
                    usedUintRegs |= (1UL << b);
                    break;

                case ComputeOpcode.Return:
                    scanTerminated = true;
                    break;
            }
        }

        Span<int> uintMap = stackalloc int[64];
        Span<int> floatMap = stackalloc int[64];
        var alloc = new PtxRegisterAllocator(uintMap, floatMap);

        // 1. Reserve ABI parameter & thread-index registers first so virtual registers never collide.
        int rdParamA = alloc.AllocRd();
        int rdParamB = alloc.AllocRd();
        int rdParamOut = alloc.AllocRd();

        int rParamN = alloc.AllocR();
        int rCtaId = alloc.AllocR();
        int rNtid = alloc.AllocR();
        int rTid = alloc.AllocR();
        int rGid = alloc.AllocR();

        int pExitGuard = alloc.AllocPred();

        // 2. Allocate dedicated non-clobbering scratch registers for memory/shuffle lowering.
        int pMemGuard = (usesGlobalMem || usesShared) ? alloc.AllocPred() : -1;
        int rShflIn = usesShuffle ? alloc.AllocR() : -1;
        int rShflOut = usesShuffle ? alloc.AllocR() : -1;
        int rdByteOffset = (usesGlobalMem || usesShared) ? alloc.AllocRd() : -1;
        int rdAddr = (usesGlobalMem || usesShared) ? alloc.AllocRd() : -1;
        int rdSmemBase = usesShared ? alloc.AllocRd() : -1;

        var bodySb = new StringBuilder(2048);

        // Parameter loading
        bodySb.AppendLine("    // Load kernel parameters");
        bodySb.Append("    ld.param.u64 %rd").Append(rdParamA).AppendLine(", [param_input_a];");
        bodySb.Append("    ld.param.u64 %rd").Append(rdParamB).AppendLine(", [param_input_b];");
        bodySb.Append("    ld.param.u64 %rd").Append(rdParamOut).AppendLine(", [param_output];");
        bodySb.Append("    ld.param.u32 %r").Append(rParamN).AppendLine(", [param_n];");
        bodySb.AppendLine();

        // Thread index calculation
        bodySb.AppendLine("    // Global thread index: gid = ctaid.x * ntid.x + tid.x");
        bodySb.Append("    mov.u32 %r").Append(rCtaId).AppendLine(", %ctaid.x;");
        bodySb.Append("    mov.u32 %r").Append(rNtid).AppendLine(", %ntid.x;");
        bodySb.Append("    mov.u32 %r").Append(rTid).AppendLine(", %tid.x;");
        bodySb.Append("    mad.lo.u32 %r").Append(rGid).Append(", %r").Append(rCtaId).Append(", %r").Append(rNtid).Append(", %r").Append(rTid).AppendLine(";");
        bodySb.AppendLine();

        // Bounds check
        bodySb.AppendLine("    // Boundary guard");
        bodySb.Append("    setp.ge.u32 %p").Append(pExitGuard).Append(", %r").Append(rGid).Append(", %r").Append(rParamN).AppendLine(";");
        bodySb.Append("    @%p").Append(pExitGuard).AppendLine(" bra KERNEL_EXIT;");
        bodySb.AppendLine();

        // Pre-allocate and zero-initialize live typed virtual registers
        if (usedUintRegs != 0 || usedFloatRegs != 0)
        {
            bodySb.AppendLine("    // Initialize live typed virtual registers");
            for (int r = 0; r < 64; r++)
            {
                if ((usedUintRegs & (1UL << r)) != 0)
                {
                    int physR = alloc.GetOrAllocUint(r);
                    bodySb.Append("    mov.u32 %r").Append(physR).AppendLine(", 0;");
                }

                if ((usedFloatRegs & (1UL << r)) != 0)
                {
                    int physF = alloc.GetOrAllocFloat(r);
                    bodySb.Append("    mov.f32 %f").Append(physF).AppendLine(", 0f00000000;");
                }
            }

            bodySb.AppendLine();
        }

        bodySb.AppendLine("    // Kernel Instruction Stream");

        bool terminated = false;
        for (int i = 0; i < instructions.Length && !terminated; i++)
        {
            var inst = instructions[i];
            var opcode = (ComputeOpcode)inst.Opcode;
            int a = Math.Clamp(inst.OperandA, 0, 63);
            int b = Math.Clamp(inst.OperandB, 0, 63);
            int d = Math.Clamp(inst.Destination, 0, 63);

            switch (opcode)
            {
                case ComputeOpcode.Nop:
                    bodySb.AppendLine("    nop;");
                    break;

                case ComputeOpcode.Add:
                    bodySb.Append("    add.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Sub:
                    bodySb.Append("    sub.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Mul:
                    bodySb.Append("    mul.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Div:
                    bodySb.Append("    div.rn.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Fma:
                    bodySb.Append("    fma.rn.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).Append(", %f").Append(alloc.GetFloat(d)).AppendLine(";");
                    break;

                case ComputeOpcode.Neg:
                    bodySb.Append("    neg.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                    break;

                case ComputeOpcode.Abs:
                    bodySb.Append("    abs.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                    break;

                case ComputeOpcode.Min:
                    bodySb.Append("    min.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Max:
                    bodySb.Append("    max.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).Append(", %f").Append(alloc.GetFloat(b)).AppendLine(";");
                    break;

                case ComputeOpcode.Sqrt:
                    bodySb.Append("    sqrt.rn.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                    break;

                case ComputeOpcode.Rsqrt:
                    bodySb.Append("    rsqrt.approx.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                    break;

                case ComputeOpcode.Exp:
                    {
                        int fA = alloc.GetFloat(a);
                        int fD = alloc.GetFloat(d);
                        bodySb.Append("    mul.f32 %f").Append(fD).Append(", %f").Append(fA).AppendLine(", 0f3FB8AA3B;");
                        bodySb.Append("    ex2.approx.f32 %f").Append(fD).Append(", %f").Append(fD).AppendLine(";");
                        break;
                    }

                case ComputeOpcode.Log:
                    {
                        int fA = alloc.GetFloat(a);
                        int fD = alloc.GetFloat(d);
                        bodySb.Append("    lg2.approx.f32 %f").Append(fD).Append(", %f").Append(fA).AppendLine(";");
                        bodySb.Append("    mul.f32 %f").Append(fD).Append(", %f").Append(fD).AppendLine(", 0f3F317218;");
                        break;
                    }

                case ComputeOpcode.Tanh:
                    bodySb.Append("    tanh.approx.f32 %f").Append(alloc.GetFloat(d)).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                    break;

                case ComputeOpcode.LoadGlobal:
                    {
                        int bufIdx = Math.Clamp(inst.OperandA, 0, 2);
                        int basePtrReg = bufIdx switch
                        {
                            1 => rdParamB,
                            2 => rdParamOut,
                            _ => rdParamA
                        };
                        int rIdx = alloc.GetUint(b);
                        int fDest = alloc.GetFloat(d);

                        bodySb.Append("    setp.ge.u32 %p").Append(pMemGuard).Append(", %r").Append(rIdx).Append(", %r").Append(rParamN).AppendLine(";");
                        bodySb.Append("    mul.wide.u32 %rd").Append(rdByteOffset).Append(", %r").Append(rIdx).AppendLine(", 4;");
                        bodySb.Append("    add.u64 %rd").Append(rdAddr).Append(", %rd").Append(basePtrReg).Append(", %rd").Append(rdByteOffset).AppendLine(";");
                        bodySb.Append("    @!%p").Append(pMemGuard).Append(" ld.global.f32 %f").Append(fDest).Append(", [%rd").Append(rdAddr).AppendLine("];");
                        bodySb.Append("    @%p").Append(pMemGuard).Append(" mov.f32 %f").Append(fDest).AppendLine(", 0f00000000;");
                        break;
                    }

                case ComputeOpcode.StoreGlobal:
                    {
                        int rIdx = alloc.GetUint(b);
                        int fSrc = alloc.GetFloat(a);

                        bodySb.Append("    setp.ge.u32 %p").Append(pMemGuard).Append(", %r").Append(rIdx).Append(", %r").Append(rParamN).AppendLine(";");
                        bodySb.Append("    mul.wide.u32 %rd").Append(rdByteOffset).Append(", %r").Append(rIdx).AppendLine(", 4;");
                        bodySb.Append("    add.u64 %rd").Append(rdAddr).Append(", %rd").Append(rdParamOut).Append(", %rd").Append(rdByteOffset).AppendLine(";");
                        bodySb.Append("    @!%p").Append(pMemGuard).Append(" st.global.f32 [%rd").Append(rdAddr).Append("], %f").Append(fSrc).AppendLine(";");
                        break;
                    }

                case ComputeOpcode.LoadShared:
                    {
                        int rIdx = alloc.GetUint(b);
                        int fDest = alloc.GetFloat(d);

                        bodySb.Append("    setp.ge.u32 %p").Append(pMemGuard).Append(", %r").Append(rIdx).AppendLine(", 1024;");
                        bodySb.Append("    mul.wide.u32 %rd").Append(rdByteOffset).Append(", %r").Append(rIdx).AppendLine(", 4;");
                        bodySb.Append("    mov.u64 %rd").Append(rdSmemBase).AppendLine(", smem_workspace;");
                        bodySb.Append("    add.u64 %rd").Append(rdAddr).Append(", %rd").Append(rdSmemBase).Append(", %rd").Append(rdByteOffset).AppendLine(";");
                        bodySb.Append("    @!%p").Append(pMemGuard).Append(" ld.shared.f32 %f").Append(fDest).Append(", [%rd").Append(rdAddr).AppendLine("];");
                        bodySb.Append("    @%p").Append(pMemGuard).Append(" mov.f32 %f").Append(fDest).AppendLine(", 0f00000000;");
                        break;
                    }

                case ComputeOpcode.StoreShared:
                    {
                        int rIdx = alloc.GetUint(b);
                        int fSrc = alloc.GetFloat(a);

                        bodySb.Append("    setp.ge.u32 %p").Append(pMemGuard).Append(", %r").Append(rIdx).AppendLine(", 1024;");
                        bodySb.Append("    mul.wide.u32 %rd").Append(rdByteOffset).Append(", %r").Append(rIdx).AppendLine(", 4;");
                        bodySb.Append("    mov.u64 %rd").Append(rdSmemBase).AppendLine(", smem_workspace;");
                        bodySb.Append("    add.u64 %rd").Append(rdAddr).Append(", %rd").Append(rdSmemBase).Append(", %rd").Append(rdByteOffset).AppendLine(";");
                        bodySb.Append("    @!%p").Append(pMemGuard).Append(" st.shared.f32 [%rd").Append(rdAddr).Append("], %f").Append(fSrc).AppendLine(";");
                        break;
                    }

                case ComputeOpcode.LoadConstant:
                    {
                        uint bits = (uint)inst.OperandA;
                        if ((usedFloatRegs & (1UL << d)) != 0)
                        {
                            bodySb.Append("    mov.f32 %f").Append(alloc.GetFloat(d)).Append(", 0f").Append(bits.ToString("X8", CultureInfo.InvariantCulture)).AppendLine(";");
                        }

                        if ((usedUintRegs & (1UL << d)) != 0)
                        {
                            bodySb.Append("    mov.u32 %r").Append(alloc.GetUint(d)).Append(", 0x").Append(bits.ToString("X8", CultureInfo.InvariantCulture)).AppendLine(";");
                        }

                        break;
                    }

                case ComputeOpcode.ShflSyncBfly:
                    {
                        int laneMask = inst.OperandB != 0 ? inst.OperandB : 16;
                        bodySb.Append("    mov.b32 %r").Append(rShflIn).Append(", %f").Append(alloc.GetFloat(a)).AppendLine(";");
                        bodySb.Append("    shfl.sync.bfly.b32 %r").Append(rShflOut).Append(", %r").Append(rShflIn).Append(", ").Append(laneMask).AppendLine(", 0x1f, 0xffffffff;");
                        bodySb.Append("    mov.b32 %f").Append(alloc.GetFloat(d)).Append(", %r").Append(rShflOut).AppendLine(";");
                        break;
                    }

                case ComputeOpcode.BarrierSync:
                    bodySb.AppendLine("    bar.sync 0;");
                    break;

                case ComputeOpcode.LocalId:
                    {
                        int rDest = alloc.GetUint(d);
                        bodySb.Append("    mov.u32 %r").Append(rDest).AppendLine(", %tid.x;");
                        if ((usedFloatRegs & (1UL << d)) != 0)
                        {
                            bodySb.Append("    cvt.rn.f32.u32 %f").Append(alloc.GetFloat(d)).Append(", %r").Append(rDest).AppendLine(";");
                        }

                        break;
                    }

                case ComputeOpcode.GlobalId:
                    {
                        int rDest = alloc.GetUint(d);
                        bodySb.Append("    mov.u32 %r").Append(rDest).Append(", %r").Append(rGid).AppendLine(";");
                        if ((usedFloatRegs & (1UL << d)) != 0)
                        {
                            bodySb.Append("    cvt.rn.f32.u32 %f").Append(alloc.GetFloat(d)).Append(", %r").Append(rDest).AppendLine(";");
                        }

                        break;
                    }

                case ComputeOpcode.AtomicAdd:
                    {
                        int rIdx = alloc.GetUint(b);
                        int fSrc = alloc.GetFloat(a);
                        int fDest = alloc.GetFloat(d);

                        bodySb.Append("    setp.ge.u32 %p").Append(pMemGuard).Append(", %r").Append(rIdx).Append(", %r").Append(rParamN).AppendLine(";");
                        bodySb.Append("    mul.wide.u32 %rd").Append(rdByteOffset).Append(", %r").Append(rIdx).AppendLine(", 4;");
                        bodySb.Append("    add.u64 %rd").Append(rdAddr).Append(", %rd").Append(rdParamOut).Append(", %rd").Append(rdByteOffset).AppendLine(";");
                        bodySb.Append("    @!%p").Append(pMemGuard).Append(" atom.global.add.f32 %f").Append(fDest).Append(", [%rd").Append(rdAddr).Append("], %f").Append(fSrc).AppendLine(";");
                        bodySb.Append("    @%p").Append(pMemGuard).Append(" mov.f32 %f").Append(fDest).AppendLine(", 0f00000000;");
                        break;
                    }

                case ComputeOpcode.Return:
                    bodySb.AppendLine("    bra KERNEL_EXIT;");
                    terminated = true;
                    break;

                default:
                    throw new NotSupportedException($"Unsupported ComputeOpcode '{inst.Opcode}' for PTX 8.0 emission.");
            }
        }

        bodySb.AppendLine();
        bodySb.AppendLine("KERNEL_EXIT:");
        bodySb.AppendLine("    ret;");
        bodySb.AppendLine("}");

        // Assemble final PTX with exact emitted register counts.
        var sb = new StringBuilder(bodySb.Length + 512);
        sb.AppendLine("// ===========================================================================");
        sb.AppendLine("// Pure C# .NET 10 PTX 8.0 JIT Generator (Glacier.Compute.Assembler)");
        sb.AppendLine("// ===========================================================================");
        sb.AppendLine(".version 8.0");
        sb.Append(".target ").AppendLine(targetArch);
        sb.AppendLine(".address_size 64");
        sb.AppendLine();

        sb.Append(".visible .entry ").Append(kernelName).AppendLine("(");
        sb.AppendLine("    .param .u64 param_input_a,");
        sb.AppendLine("    .param .u64 param_input_b,");
        sb.AppendLine("    .param .u64 param_output,");
        sb.AppendLine("    .param .u32 param_n");
        sb.AppendLine(")");
        sb.AppendLine("{");

        sb.Append("    .reg .pred %p<").Append(alloc.PredCount).AppendLine(">;");
        sb.Append("    .reg .b32  %r<").Append(alloc.RCount).AppendLine(">;");
        sb.Append("    .reg .b64  %rd<").Append(alloc.RdCount).AppendLine(">;");
        if (alloc.FCount > 0)
        {
            sb.Append("    .reg .f32  %f<").Append(alloc.FCount).AppendLine(">;");
        }

        if (alloc.FdCount > 0)
        {
            sb.Append("    .reg .f64  %fd<").Append(alloc.FdCount).AppendLine(">;");
        }

        if (usesShared)
        {
            sb.AppendLine("    .shared .align 4 .b8 smem_workspace[4096];");
        }

        sb.AppendLine();
        sb.Append(bodySb);
        return sb.ToString();
    }

    private ref struct PtxRegisterAllocator
    {
        private readonly Span<int> _uintMap;
        private readonly Span<int> _floatMap;
        private int _predCount;
        private int _rCount;
        private int _rdCount;
        private int _fCount;
        private int _fdCount;

        public PtxRegisterAllocator(Span<int> uintMap, Span<int> floatMap)
        {
            _uintMap = uintMap;
            _floatMap = floatMap;
            _uintMap.Fill(-1);
            _floatMap.Fill(-1);
        }

        public readonly int PredCount => _predCount;
        public readonly int RCount => _rCount;
        public readonly int RdCount => _rdCount;
        public readonly int FCount => _fCount;
        public readonly int FdCount => _fdCount;

        public int AllocPred() => _predCount++;
        public int AllocR() => _rCount++;
        public int AllocRd() => _rdCount++;
        public int AllocF() => _fCount++;
        public int AllocFd() => _fdCount++;

        public int GetOrAllocUint(int vreg)
        {
            int idx = _uintMap[vreg];
            if (idx < 0)
            {
                idx = AllocR();
                _uintMap[vreg] = idx;
            }

            return idx;
        }

        public int GetOrAllocFloat(int vreg)
        {
            int idx = _floatMap[vreg];
            if (idx < 0)
            {
                idx = AllocF();
                _floatMap[vreg] = idx;
            }

            return idx;
        }

        public readonly int GetUint(int vreg)
        {
            int idx = _uintMap[vreg];
            if (idx < 0)
            {
                throw new InvalidOperationException($"Virtual uint register r{vreg} was referenced without being allocated.");
            }

            return idx;
        }

        public readonly int GetFloat(int vreg)
        {
            int idx = _floatMap[vreg];
            if (idx < 0)
            {
                throw new InvalidOperationException($"Virtual float register f{vreg} was referenced without being allocated.");
            }

            return idx;
        }
    }
}

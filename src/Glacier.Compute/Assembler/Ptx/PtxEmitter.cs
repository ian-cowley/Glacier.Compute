// <copyright file="PtxEmitter.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Ptx;

using System;
using System.Text;

/// <summary>
/// High-performance NVIDIA PTX 8.0 JIT assembly generator.
/// Generates compliant PTX 8.0 text for sm_80, sm_89, and sm_90 architectures.
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
        var sb = new StringBuilder(4096);
        sb.AppendLine("// ===========================================================================");
        sb.AppendLine("// Pure C# .NET 10 PTX 8.0 JIT Generator (Glacier.Compute.Assembler)");
        sb.AppendLine("// ===========================================================================");
        sb.AppendLine(".version 8.0");
        sb.Append(".target ").AppendLine(targetArch);
        sb.AppendLine(".address_size 64");
        sb.AppendLine();

        // Kernel signature
        sb.Append(".visible .entry ").Append(kernelName).AppendLine("(");
        sb.AppendLine("    .param .u64 param_input_a,");
        sb.AppendLine("    .param .u64 param_input_b,");
        sb.AppendLine("    .param .u64 param_output,");
        sb.AppendLine("    .param .u32 param_n");
        sb.AppendLine(")");
        sb.AppendLine("{");

        // Register allocations
        sb.AppendLine("    .reg .pred %p<16>;");
        sb.AppendLine("    .reg .b32  %r<64>;");
        sb.AppendLine("    .reg .b64  %rd<32>;");
        sb.AppendLine("    .reg .f32  %f<64>;");
        sb.AppendLine("    .shared .align 4 .b8 smem_workspace[4096];");
        sb.AppendLine();

        // Parameter loading
        sb.AppendLine("    // Load kernel parameters");
        sb.AppendLine("    ld.param.u64 %rd1, [param_input_a];");
        sb.AppendLine("    ld.param.u64 %rd2, [param_input_b];");
        sb.AppendLine("    ld.param.u64 %rd3, [param_output];");
        sb.AppendLine("    ld.param.u32 %r1,  [param_n];");
        sb.AppendLine();

        // Calculate thread indices
        sb.AppendLine("    // Global thread index: gid = ctaid.x * ntid.x + tid.x");
        sb.AppendLine("    mov.u32 %r2, %ctaid.x;");
        sb.AppendLine("    mov.u32 %r3, %ntid.x;");
        sb.AppendLine("    mov.u32 %r4, %tid.x;");
        sb.AppendLine("    mad.lo.u32 %r5, %r2, %r3, %r4;");
        sb.AppendLine();

        // Bounds check
        sb.AppendLine("    // Boundary guard");
        sb.AppendLine("    setp.ge.u32 %p1, %r5, %r1;");
        sb.AppendLine("    @%p1 bra KERNEL_EXIT;");
        sb.AppendLine();

        sb.AppendLine("    // Kernel Instruction Stream");

        for (int i = 0; i < instructions.Length; i++)
        {
            var inst = instructions[i];
            var opcode = (ComputeOpcode)inst.Opcode;
            int a = Math.Clamp(inst.OperandA, 0, 63);
            int b = Math.Clamp(inst.OperandB, 0, 63);
            int d = Math.Clamp(inst.Destination, 0, 63);

            switch (opcode)
            {
                case ComputeOpcode.Nop:
                    sb.AppendLine("    nop;");
                    break;

                case ComputeOpcode.Add:
                    sb.Append("    add.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Sub:
                    sb.Append("    sub.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Mul:
                    sb.Append("    mul.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Div:
                    sb.Append("    div.rn.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Fma:
                    sb.Append("    fma.rn.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).Append(", %f").Append(d).AppendLine(";");
                    break;

                case ComputeOpcode.Neg:
                    sb.Append("    neg.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Abs:
                    sb.Append("    abs.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Min:
                    sb.Append("    min.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Max:
                    sb.Append("    max.f32 %f").Append(d).Append(", %f").Append(a).Append(", %f").Append(b).AppendLine(";");
                    break;

                case ComputeOpcode.Sqrt:
                    sb.Append("    sqrt.rn.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Rsqrt:
                    sb.Append("    rsqrt.approx.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Exp:
                    sb.Append("    ex2.approx.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Log:
                    sb.Append("    lg2.approx.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Tanh:
                    sb.Append("    tanh.approx.f32 %f").Append(d).Append(", %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.LoadGlobal:
                    sb.Append("    mul.wide.u32 %rd10, %r").Append(b).AppendLine(", 4;");
                    sb.AppendLine("    add.u64 %rd11, %rd1, %rd10;");
                    sb.Append("    ld.global.f32 %f").Append(d).AppendLine(", [%rd11];");
                    break;

                case ComputeOpcode.StoreGlobal:
                    sb.Append("    mul.wide.u32 %rd12, %r").Append(b).AppendLine(", 4;");
                    sb.AppendLine("    add.u64 %rd13, %rd3, %rd12;");
                    sb.Append("    st.global.f32 [%rd13], %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.LoadShared:
                    sb.Append("    mul.wide.u32 %rd14, %r").Append(b).AppendLine(", 4;");
                    sb.AppendLine("    mov.u64 %rd15, smem_workspace;");
                    sb.AppendLine("    add.u64 %rd16, %rd15, %rd14;");
                    sb.Append("    ld.shared.f32 %f").Append(d).AppendLine(", [%rd16];");
                    break;

                case ComputeOpcode.StoreShared:
                    sb.Append("    mul.wide.u32 %rd17, %r").Append(b).AppendLine(", 4;");
                    sb.AppendLine("    mov.u64 %rd18, smem_workspace;");
                    sb.AppendLine("    add.u64 %rd19, %rd18, %rd17;");
                    sb.Append("    st.shared.f32 [%rd19], %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.LoadConstant:
                    sb.Append("    mov.b32 %f").Append(d).Append(", ").Append(BitConverter.UInt32BitsToSingle((uint)inst.OperandA).ToString("R", System.Globalization.CultureInfo.InvariantCulture)).AppendLine(";");
                    break;

                case ComputeOpcode.ShflSyncBfly:
                    int laneMask = inst.OperandB != 0 ? inst.OperandB : 16;
                    sb.Append("    shfl.sync.bfly.b32 %f").Append(d).Append(", %f").Append(a).Append(", ").Append(laneMask).AppendLine(", 0x1f, 0xffffffff;");
                    break;

                case ComputeOpcode.BarrierSync:
                    sb.AppendLine("    bar.sync 0;");
                    break;

                case ComputeOpcode.LocalId:
                    sb.Append("    mov.u32 %r").Append(d).AppendLine(", %tid.x;");
                    break;

                case ComputeOpcode.GlobalId:
                    sb.Append("    mov.u32 %r").Append(d).AppendLine(", %r5;");
                    break;

                case ComputeOpcode.AtomicAdd:
                    sb.Append("    mul.wide.u32 %rd20, %r").Append(b).AppendLine(", 4;");
                    sb.AppendLine("    add.u64 %rd21, %rd3, %rd20;");
                    sb.Append("    atom.global.add.f32 %f").Append(d).Append(", [%rd21], %f").Append(a).AppendLine(";");
                    break;

                case ComputeOpcode.Return:
                    sb.AppendLine("    bra KERNEL_EXIT;");
                    break;

                default:
                    sb.Append("    // Unsupported opcode: ").AppendLine(inst.Opcode.ToString());
                    break;
            }
        }

        sb.AppendLine();
        sb.AppendLine("KERNEL_EXIT:");
        sb.AppendLine("    ret;");
        sb.AppendLine("}");
        return sb.ToString();
    }
}

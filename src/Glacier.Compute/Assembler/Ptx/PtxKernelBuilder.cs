// <copyright file="PtxKernelBuilder.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Ptx;

using System;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// Fluent programmatic builder for NVIDIA PTX 8.0 assembly compute kernels.
/// </summary>
public sealed class PtxKernelBuilder
{
    private readonly string _kernelName;
    private string _targetArch = "sm_90";
    private readonly List<string> _parameters = new();
    private readonly List<string> _bodyLines = new();
    private int _reg32Count = 32;
    private int _reg64Count = 32;
    private int _regFp32Count = 32;
    private int _sharedMemBytes = 4096;

    /// <summary>
    /// Initializes a new instance of the <see cref="PtxKernelBuilder"/> class.
    /// </summary>
    /// <param name="kernelName">Name of the entry kernel.</param>
    public PtxKernelBuilder(string kernelName = "glacier_kernel")
    {
        _kernelName = kernelName;
    }

    /// <summary>Sets the target architecture (e.g. sm_80, sm_89, sm_90).</summary>
    public PtxKernelBuilder SetTargetArch(string arch)
    {
        _targetArch = arch;
        return this;
    }

    /// <summary>Adds a kernel parameter declaration.</summary>
    public PtxKernelBuilder AddParameter(string type, string name)
    {
        _parameters.Add($"{type} {name}");
        return this;
    }

    /// <summary>Configures the register pool count.</summary>
    public PtxKernelBuilder SetRegisterCounts(int b32 = 32, int b64 = 32, int f32 = 32)
    {
        _reg32Count = b32;
        _reg64Count = b64;
        _regFp32Count = f32;
        return this;
    }

    /// <summary>Configures workgroup shared memory allocation in bytes.</summary>
    public PtxKernelBuilder SetSharedMemoryBytes(int bytes)
    {
        _sharedMemBytes = bytes;
        return this;
    }

    /// <summary>Appends a raw PTX instruction line.</summary>
    public PtxKernelBuilder Emit(string instruction)
    {
        _bodyLines.Add(instruction);
        return this;
    }

    /// <summary>Emits a warp shuffle butterfly primitive.</summary>
    public PtxKernelBuilder EmitShuffleBfly(string dstReg, string srcReg, int laneMask)
    {
        _bodyLines.Add($"shfl.sync.bfly.b32 {dstReg}, {srcReg}, {laneMask}, 0x1f, 0xffffffff;");
        return this;
    }

    /// <summary>Emits a barrier synchronization.</summary>
    public PtxKernelBuilder EmitBarrier()
    {
        _bodyLines.Add("bar.sync 0;");
        return this;
    }

    /// <summary>Emits an FMA instruction.</summary>
    public PtxKernelBuilder EmitFma(string dstReg, string aReg, string bReg, string cReg)
    {
        _bodyLines.Add($"fma.rn.f32 {dstReg}, {aReg}, {bReg}, {cReg};");
        return this;
    }

    /// <summary>Builds the complete PTX 8.0 assembly source string.</summary>
    public string Build()
    {
        var sb = new StringBuilder(4096);
        sb.AppendLine(".version 8.0");
        sb.Append(".target ").AppendLine(_targetArch);
        sb.AppendLine(".address_size 64");
        sb.AppendLine();

        sb.Append(".visible .entry ").Append(_kernelName).AppendLine("(");
        for (int i = 0; i < _parameters.Count; i++)
        {
            sb.Append("    .param ").Append(_parameters[i]);
            if (i < _parameters.Count - 1)
            {
                sb.AppendLine(",");
            }
            else
            {
                sb.AppendLine();
            }
        }
        sb.AppendLine(")");
        sb.AppendLine("{");

        sb.AppendLine($"    .reg .pred %p<16>;");
        sb.AppendLine($"    .reg .b32  %r<{_reg32Count}>;");
        sb.AppendLine($"    .reg .b64  %rd<{_reg64Count}>;");
        sb.AppendLine($"    .reg .f32  %f<{_regFp32Count}>;");

        if (_sharedMemBytes > 0)
        {
            sb.AppendLine($"    .shared .align 4 .b8 smem[{_sharedMemBytes}];");
        }
        sb.AppendLine();

        foreach (var line in _bodyLines)
        {
            sb.Append("    ").AppendLine(line);
        }

        sb.AppendLine("    ret;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <inheritdoc/>
    public override string ToString() => Build();
}

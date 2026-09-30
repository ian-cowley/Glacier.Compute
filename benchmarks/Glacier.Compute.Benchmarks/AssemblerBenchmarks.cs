// <copyright file="AssemblerBenchmarks.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Assembler.SpirV;
using Glacier.Compute.Pipeline;

[MemoryDiagnoser]
public class AssemblerBenchmarks
{
    private Instruction[] _instructions = null!;
    private SpirvBinaryEmitter _spirvEmitter = null!;
    private uint[] _spirvWordBuffer = null!;
    private ShaderPipeline _pipeline = null!;

    [GlobalSetup]
    public void Setup()
    {
        _instructions = new Instruction[32];
        for (int i = 0; i < 32; i++)
        {
            _instructions[i] = new Instruction((int)ComputeOpcode.Add, i, (i + 1) % 32, (i + 2) % 32);
        }

        _spirvEmitter = new SpirvBinaryEmitter();
        _spirvWordBuffer = new uint[2048];
        _pipeline = new ShaderPipeline();
    }

    [Benchmark]
    public string EmitPtx80()
    {
        return PtxEmitter.Emit(_instructions);
    }

    [Benchmark]
    public ReadOnlyMemory<byte> EmitSpirv16()
    {
        return _spirvEmitter.EmitModule(ShaderStage.Compute, _instructions);
    }

    [Benchmark]
    public int EmitSpirv16_ZeroAllocSpan()
    {
        return _spirvEmitter.EmitWords(ShaderStage.Compute, _instructions, _spirvWordBuffer);
    }

    [Benchmark]
    public ReadOnlyMemory<byte> EmitDxbcContainer()
    {
        return DxbcContainerBuilder.BuildComputeShader(ShaderStage.Compute, _instructions);
    }

    [Benchmark]
    public ComputeKernel PipelineCachedCompile()
    {
        return _pipeline.Compile("bench_kernel", ShaderStage.Compute, _instructions);
    }
}

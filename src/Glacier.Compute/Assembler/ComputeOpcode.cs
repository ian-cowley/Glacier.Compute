// <copyright file="ComputeOpcode.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler;

/// <summary>
/// Intermediate representation opcodes for Glacier.Compute shader generation.
/// </summary>
public enum ComputeOpcode : int
{
    /// <summary>No-operation.</summary>
    Nop = 0,

    /// <summary>Floating-point or integer addition: D = A + B.</summary>
    Add = 1,

    /// <summary>Floating-point or integer subtraction: D = A - B.</summary>
    Sub = 2,

    /// <summary>Floating-point or integer multiplication: D = A * B.</summary>
    Mul = 3,

    /// <summary>Floating-point or integer division: D = A / B.</summary>
    Div = 4,

    /// <summary>Fused multiply-add: D = (A * B) + D.</summary>
    Fma = 5,

    /// <summary>Unary negation: D = -A.</summary>
    Neg = 6,

    /// <summary>Absolute value: D = abs(A).</summary>
    Abs = 7,

    /// <summary>Minimum: D = min(A, B).</summary>
    Min = 8,

    /// <summary>Maximum: D = max(A, B).</summary>
    Max = 9,

    /// <summary>Square root: D = sqrt(A).</summary>
    Sqrt = 10,

    /// <summary>Reciprocal square root: D = 1 / sqrt(A).</summary>
    Rsqrt = 11,

    /// <summary>Exponential: D = exp(A).</summary>
    Exp = 12,

    /// <summary>Natural logarithm: D = log(A).</summary>
    Log = 13,

    /// <summary>Hyperbolic tangent: D = tanh(A).</summary>
    Tanh = 14,

    /// <summary>Load from global memory buffer at index B into register D.</summary>
    LoadGlobal = 15,

    /// <summary>Store value A into global memory buffer at index B.</summary>
    StoreGlobal = 16,

    /// <summary>Load from workgroup shared memory at offset B into register D.</summary>
    LoadShared = 17,

    /// <summary>Store value A into workgroup shared memory at offset B.</summary>
    StoreShared = 18,

    /// <summary>Load immediate float / int constant into register D.</summary>
    LoadConstant = 19,

    /// <summary>Warp shuffle butterfly sync primitive: D = shfl.sync.bfly(A, B, mask=0xffffffff).</summary>
    ShflSyncBfly = 20,

    /// <summary>Workgroup barrier synchronization: bar.sync / OpControlBarrier.</summary>
    BarrierSync = 21,

    /// <summary>Retrieve thread local invocation ID X: D = LocalId.X.</summary>
    LocalId = 22,

    /// <summary>Retrieve global thread invocation ID X: D = GlobalId.X.</summary>
    GlobalId = 23,

    /// <summary>Atomic floating-point addition in global memory: [B] += A.</summary>
    AtomicAdd = 24,

    /// <summary>Kernel return / exit.</summary>
    Return = 25
}

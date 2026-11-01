// <copyright file="Contracts.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute;

using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Specifies the target pipeline stage for shader compilation.
/// </summary>
public enum ShaderStage : byte
{
    /// <summary>Compute shader pipeline stage.</summary>
    Compute = 0,

    /// <summary>Vertex shader pipeline stage.</summary>
    Vertex = 1,

    /// <summary>Pixel / fragment shader pipeline stage.</summary>
    Pixel = 2
}

/// <summary>
/// Supported reduction operators for distributed all-reduce collective operations.
/// </summary>
public enum ReductionOp : byte
{
    /// <summary>Sum reduction (addition).</summary>
    Sum = 0,

    /// <summary>Average reduction (sum divided by world size).</summary>
    Average = 1,

    /// <summary>Element-wise minimum reduction.</summary>
    Min = 2,

    /// <summary>Element-wise maximum reduction.</summary>
    Max = 3
}

/// <summary>
/// Memory tier classification for multi-tier arena sub-allocation.
/// </summary>
public enum MemoryCategory : byte
{
    /// <summary>Tier 1: Fixed-size small slabs (256 B – 64 KB).</summary>
    SlabSmall = 0,

    /// <summary>Tier 2: Buddy power-of-two blocks (128 KB – 32 MB).</summary>
    BuddyMedium = 1,

    /// <summary>Tier 3: Large virtual address reservoir or tail arena (&gt; 32 MB).</summary>
    LargeVirtual = 2
}

/// <summary>
/// Represents a typed sub-allocated slice address within an external device arena (CUDA/D3D12/Vulkan VRAM or pinned memory)
/// or an aligned unmanaged host arena, paired with its explicit byte capacity.
/// </summary>
/// <param name="Address">The native device virtual address or unmanaged host pointer.</param>
/// <param name="SizeInBytes">The capacity in bytes of the allocated region.</param>
public readonly record struct DevicePointer(IntPtr Address, ulong SizeInBytes)
{
    /// <summary>Gets a value indicating whether this pointer is null or zero-sized.</summary>
    public bool IsNull => Address == IntPtr.Zero || SizeInBytes == 0;

    /// <summary>Returns a null pointer.</summary>
    public static DevicePointer Null => new(IntPtr.Zero, 0);
}

/// <summary>
/// Low-level compute instruction representation.
/// </summary>
/// <param name="Opcode">The numeric operation code (<see cref="Assembler.ComputeOpcode"/>).</param>
/// <param name="OperandA">The first operand register or literal index.</param>
/// <param name="OperandB">The second operand register or literal index.</param>
/// <param name="Destination">The target destination register index.</param>
public readonly record struct Instruction(int Opcode, int OperandA, int OperandB, int Destination);

/// <summary>
/// Builder contract for in-memory shader bytecode emission across PTX 8.0 and SPIR-V 1.6.
/// Direct LLVM 3.7 DXIL emission from <see cref="Instruction"/> streams is quarantined because retail D3D12 drivers
/// require external DXC compilation and <c>dxil.dll</c> validator signing.
/// </summary>
public interface IShaderBytecodeBuilder
{
    /// <summary>
    /// Builds binary SPIR-V 1.6 bytecode for the specified shader stage and instructions.
    /// </summary>
    ReadOnlyMemory<byte> BuildSpirV(ShaderStage stage, ReadOnlySpan<Instruction> instructions);

    /// <summary>
    /// Builds a DirectX Container (DXBC) containing DXIL bitcode for Direct3D 12, if supported by the implementation.
    /// The default pure-C# builder throws <see cref="NotSupportedException"/> because retail D3D12 drivers reject unsigned
    /// non-DXC LLVM 3.7 bitcode; use <see cref="Assembler.Dxbc.DxbcContainerBuilder.PackagePrecompiledShader"/> for precompiled blobs.
    /// </summary>
    ReadOnlyMemory<byte> BuildDxil(ShaderStage stage, ReadOnlySpan<Instruction> instructions);

    /// <summary>
    /// Compiles NVIDIA PTX 8.0 assembly text for the specified instructions.
    /// </summary>
    string BuildPtx(ReadOnlySpan<Instruction> instructions);
}

/// <summary>
/// Multi-tier arena sub-allocator contract capable of managing either caller-supplied GPU VRAM / pinned device arenas
/// (e.g. <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, <c>ID3D12Heap</c>) or standalone unmanaged host memory arenas.
/// </summary>
public interface IVramAllocator : IDisposable
{
    /// <summary>
    /// Sub-allocates memory of the requested size and category from the backing arena.
    /// </summary>
    DevicePointer Allocate(ulong byteCount, MemoryCategory category);

    /// <summary>
    /// Frees an allocated arena slice.
    /// </summary>
    void Free(DevicePointer ptr);

    /// <summary>
    /// Rents a temporary scratch workspace from the circular ring buffer.
    /// </summary>
    ScratchWorkspace RentScratchWorkspace(ulong minimumBytes);
}

/// <summary>
/// Distributed context contract for collective operations across cluster ranks.
/// </summary>
public interface IDistributedContext : IDisposable
{
    /// <summary>Gets the current node rank (0 to WorldSize - 1).</summary>
    int Rank { get; }

    /// <summary>Gets the total cluster size (number of nodes in the ring).</summary>
    int WorldSize { get; }

    /// <summary>
    /// Performs an asynchronous Ring AllReduce over the provided buffer.
    /// </summary>
    ValueTask AllReduceAsync(Memory<float> buffer, ReductionOp op, CancellationToken ct = default);
}

/// <summary>
/// High-speed scoped scratch workspace rented from a circular ring buffer.
/// Disposing returns the slice to the allocator without driver overhead.
/// </summary>
public readonly ref struct ScratchWorkspace
{
    /// <summary>The base native pointer of the rented workspace.</summary>
    public readonly IntPtr Pointer;

    /// <summary>The allocated byte capacity of the workspace.</summary>
    public readonly ulong Capacity;

    private readonly IVramAllocator _allocator;

    /// <summary>
    /// Initializes a new instance of the <see cref="ScratchWorkspace"/> struct.
    /// </summary>
    public ScratchWorkspace(IntPtr ptr, ulong capacity, IVramAllocator allocator)
    {
        Pointer = ptr;
        Capacity = capacity;
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
    }

    /// <summary>
    /// Returns the scratch workspace to the allocator.
    /// </summary>
    public void Dispose() => _allocator.Free(new DevicePointer(Pointer, Capacity));
}

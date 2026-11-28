# Changelog

## v1.1.0

- **SPIR-V 1.6 Compute Module Emission & Validation:** Added `StorageBuffer` descriptor bindings (`set=0, binding=0..2`), `PushConstant` (`param_n`), `Workgroup` shared memory, `GLSL.std.450` math instructions (`Fma`, `FAbs`, `FMin`, `FMax`, `Sqrt`, `InverseSqrt`, `Exp`, `Log`, `Tanh`), `OpGroupNonUniformShuffleXor`, `OpAtomicFAddEXT`, compute-stage built-in guards, single-block CFG termination, and `SpirvValidator` bounds checks.
- **Pre-Compiled DXBC Packaging & DXIL Boundary:** Added `DxbcContainerBuilder.PackagePrecompiledShader` and `DxilChunkEmitter.CreateDxilChunkFromBitcode` for precompiled DXC LLVM 3.7 bitcode payloads; gated un-signed direct IR-to-DXIL synthesis behind `NotSupportedException` (`ShaderPipeline.Compile` defaults `includeDxil: false`).
- **External Device Arena Sub-Allocation:** Added `(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction)` constructors on `DeviceMemoryPool`, `SlabAllocator`, `BuddyAllocator`, `VirtualMemoryReservoir`, and `RingScratchWorkspace` so callers can sub-allocate CUDA, Direct3D 12, or Vulkan device arenas without CPU pointer dereferences.
- **Lock-Free CAS `RingScratchWorkspace`:** Switched wrap-padding coordination to an atomic `Interlocked.CompareExchange` loop to prevent concurrent slice overlap.
- **Zero-RSS Unix `mmap` Reservoir & Lazy Slabs:** Implemented `mmap(PROT_NONE)` + `mprotect` + `madvise(MADV_DONTNEED)` on Linux/macOS and lazy per-bin allocation in `SlabAllocator`.
- **Native AOT & Warning Hygiene:** Enabled `<IsAotCompatible>true</IsAotCompatible>` and `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`, moving `CuDriver` bindings to `delegate* unmanaged` function pointers and `[LibraryImport]` P/Invokes.

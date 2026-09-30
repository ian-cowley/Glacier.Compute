// <copyright file="DxbcChunk.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Assembler.Dxbc;

using System;
using System.Text;

/// <summary>
/// Represents an individual chunk in a DirectX Container (DXBC).
/// </summary>
public sealed class DxbcChunk
{
    /// <summary>FourCC for DXIL bitcode chunk ('DXIL').</summary>
    public const uint FourCcDxil = 0x4C495844;

    /// <summary>FourCC for Pipeline State Validation chunk ('PSV0').</summary>
    public const uint FourCcPsv0 = 0x30565350;

    /// <summary>FourCC for Container Hash chunk ('HASH').</summary>
    public const uint FourCcHash = 0x48534148;

    /// <summary>FourCC for Input Signature chunk ('ISG1').</summary>
    public const uint FourCcIsg1 = 0x31475349;

    /// <summary>FourCC for Root Signature chunk ('RTS0').</summary>
    public const uint FourCcRts0 = 0x30535452;

    /// <summary>Gets the 4-byte FourCC identifier.</summary>
    public uint FourCc { get; }

    /// <summary>Gets the ASCII string representation of the FourCC.</summary>
    public string FourCcString => Encoding.ASCII.GetString(BitConverter.GetBytes(FourCc));

    /// <summary>Gets the raw payload data of the chunk.</summary>
    public byte[] Data { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="DxbcChunk"/> class.
    /// </summary>
    public DxbcChunk(uint fourCc, byte[] data)
    {
        FourCc = fourCc;
        Data = data ?? Array.Empty<byte>();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DxbcChunk"/> class using an ASCII FourCC name.
    /// </summary>
    public DxbcChunk(string fourCcName, byte[] data)
    {
        if (fourCcName == null || fourCcName.Length != 4)
        {
            throw new ArgumentException("FourCC name must be exactly 4 characters.", nameof(fourCcName));
        }

        byte[] chars = Encoding.ASCII.GetBytes(fourCcName);
        FourCc = BitConverter.ToUInt32(chars, 0);
        Data = data ?? Array.Empty<byte>();
    }
}

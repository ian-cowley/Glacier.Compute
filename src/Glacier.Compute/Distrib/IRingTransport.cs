// <copyright file="IRingTransport.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Compute.Distrib;

using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Transport layer contract for distributed ring point-to-point exchanges.
/// </summary>
public interface IRingTransport : IDisposable
{
    /// <summary>Gets the local node rank.</summary>
    int Rank { get; }

    /// <summary>Gets the cluster world size.</summary>
    int WorldSize { get; }

    /// <summary>
    /// Asynchronously transmits a float tensor slice to the successor node.
    /// </summary>
    ValueTask SendToSuccessorAsync(ReadOnlyMemory<float> data, CancellationToken ct = default);

    /// <summary>
    /// Asynchronously receives a float tensor slice from the predecessor node.
    /// </summary>
    ValueTask ReceiveFromPredecessorAsync(Memory<float> buffer, CancellationToken ct = default);
}

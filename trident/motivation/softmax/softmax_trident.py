# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trident-adapted copy of ``ops/softmax.py`` (forward only).

Keeps gems host/kernel branching and the ``softmax`` / ``softmax_out`` split.
Strips ``@libentry`` / ``@triton.heuristics`` (tiles computed on host) so
``@trident.jit`` can capture the host. No backward.
"""

import torch
import trident
import triton
import triton.language as tl

from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn


@triton.jit
def softmax_kernel_non_inner_trident(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def softmax_kernel_inner_trident(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


def _tile_n_inner(N: int) -> int:
    # mirrors softmax_heur_tile_n_inner
    if N <= (32 * 1024):
        return triton.next_power_of_2(N)
    return 4096


def _num_warps_inner(TILE_N: int) -> int:
    # mirrors softmax_heur_num_warps_inner
    if TILE_N < 2048:
        return 4
    if TILE_N < 4096:
        return 8
    return 16


def _tile_k_non_inner(M: int, K: int) -> int:
    # mirrors softmax_heur_tile_k; integer waves to stay Trident/Dynamo-friendly
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    tile_k = 1
    upper_bound = min(K, MAX_TILE_K)
    while tile_k * 2 <= upper_bound:
        num_blocks = M * triton.cdiv(K, tile_k)
        # equivalent to (num_blocks / NUM_SMS) > 1 without float division
        if num_blocks > NUM_SMS:
            tile_k *= 2
        else:
            break
    return tile_k


def _tile_n_non_inner(TILE_K: int) -> int:
    # mirrors softmax_heur_tile_n_non_inner
    return triton.cdiv(8192, TILE_K)


def _num_warps_non_inner(TILE_N: int, TILE_K: int) -> int:
    # mirrors softmax_heur_num_warps_non_inner
    tile_size = TILE_N * TILE_K
    if tile_size < 2048:
        return 4
    if tile_size < 4096:
        return 8
    return 16


def softmax_out_trident(input, dim=-1, half_to_float=False, *, out):
    """Same host logic as gems ``softmax_out`` (forward only, no libentry/heuristics).

    First arg is named ``input`` (not ``self``) so ``@trident.jit`` callers stay
    compatible with torch.export binding.
    """
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"

    if input.numel() == 0:
        if tuple(out.shape) != tuple(input.shape):
            out.resize_(input.shape)
        zero_(out)
        return out

    dim = dim % input.ndim
    M = 1
    N = input.shape[dim]
    for i in range(dim):
        M *= input.shape[i]
    x = input.contiguous()
    dtype = torch.float32 if half_to_float else x.dtype
    if tuple(out.shape) != tuple(x.shape):
        out.resize_(x.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")
    K = x.numel() // M // N

    with torch_device_fn.device(x.device):
        if K > 1:
            TILE_K = _tile_k_non_inner(M, K)
            TILE_N = _tile_n_non_inner(TILE_K)
            ONE_TILE_PER_CTA = TILE_N >= N
            num_warps = _num_warps_non_inner(TILE_N, TILE_K)
            grid = (M, triton.cdiv(K, TILE_K), 1)
            softmax_kernel_non_inner_trident[grid](
                out,
                x,
                M,
                N,
                K,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                ONE_TILE_PER_CTA=ONE_TILE_PER_CTA,
                num_warps=num_warps,
            )
        else:
            TILE_N = _tile_n_inner(N)
            ONE_TILE_PER_CTA = TILE_N >= N
            num_warps = _num_warps_inner(TILE_N)
            grid = (M, 1, 1)
            softmax_kernel_inner_trident[grid](
                out,
                x,
                M,
                N,
                TILE_N=TILE_N,
                ONE_TILE_PER_CTA=ONE_TILE_PER_CTA,
                num_warps=num_warps,
            )
    return out


def _softmax_forward(input, dim=-1, half_to_float=False):
    """Shared body of gems ``softmax`` (allocates out, then ``softmax_out``)."""
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"

    if input.numel() == 0:
        out_shape = list(input.shape)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else input.dtype
    out = torch.empty_like(input, dtype=dtype)
    return softmax_out_trident(input, dim, half_to_float, out=out)


@trident.jit
def softmax_jit(input: torch.Tensor, dim: int = -1, half_to_float: bool = False) -> torch.Tensor:
    """Trident host entry (same role as ``add_jit``)."""
    return _softmax_forward(input, dim, half_to_float)


def softmax_compile_entry(
    input: torch.Tensor, dim: int = -1, half_to_float: bool = False
) -> torch.Tensor:
    """Undecorated host for ``torch.compile`` (same role as ``add_compile_entry``)."""
    return _softmax_forward(input, dim, half_to_float)


def softmax_triton(
    input: torch.Tensor, dim: int = -1, half_to_float: bool = False
) -> torch.Tensor:
    """Plain Triton host (same role as ``add_triton``)."""
    return _softmax_forward(input, dim, half_to_float)


# Aliases kept for earlier smoke tests / naming.
softmax_trident = softmax_jit


def main() -> None:
    torch.manual_seed(0)
    # One core-shape-style case first (UnaryReductionBenchmark uses dim=1).
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.softmax(x, dim=1)
    torch.testing.assert_close(softmax_triton(x, 1), ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(softmax_jit(x, 1), ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(softmax_compile_entry(x, 1), ref, atol=1e-4, rtol=1e-3)
    print("softmax_trident smoke OK", tuple(x.shape), "dim=1")


if __name__ == "__main__":
    main()

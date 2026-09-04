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

"""Trident-adapted copy of ``ops/rms_norm.py`` (forward only).

Keeps gems host branching (``N <= 4096`` one-shot vs loop) and the public
``rms_norm`` / ``rms_norm_out`` signatures for later model integration.
Strips ``@libentry`` / ``@triton.autotune`` / backward so ``@trident.jit``
can capture the host.

One-shot kernel uses the same contiguous ``pid * N + col`` addressing as the
loop path (gems always ``contiguous()`` before launch). Avoids passing
literal stride ``1`` scalars that currently trip a Trident FX-importer bug.
"""

import torch
import trident
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.jit
def rms_norm_kernel_trident(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Contiguous one-shot path (gems ``N <= 4096``)."""
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + pid * N + cols, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Cast x_normed back to input dtype before multiplying with weight
    # to align with vLLM native: x.to(weight.dtype) * weight
    x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
    y = x_normed * w
    tl.store(out_ptr + pid * N + cols, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


@triton.jit
def rms_norm_loop_kernel_trident(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)

    # Pass 1: compute sum(x^2) in chunks
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    # last step with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    # Pass 2: normalize in reverse order (better L2 cache reuse)
    prev_multiple = prev_multiple_of(N, TILE_N)

    # first reverse step with mask
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y)


def _tile_n_loop(N: int) -> int:
    # mirrors nvidia rms_norm_loop tune candidates without autotune
    if N <= 1024:
        return 1024
    if N <= 2048:
        return 2048
    if N <= 4096:
        return 4096
    return 8192


def _num_warps_loop(TILE_N: int) -> int:
    # mirrors rms_norm_loop warps candidates
    if TILE_N < 2048:
        return 4
    if TILE_N < 4096:
        return 8
    return 16


def _prod_shape(shape) -> int:
    # integer product without math.prod (Dynamo-friendly)
    out = 1
    for s in shape:
        out *= int(s)
    return out


def rms_norm_out(result, x, normalized_shape, weight, eps=1e-5):
    """Same signature as gems ``rms_norm_out`` (forward only)."""
    y = _rms_norm_forward(x, normalized_shape, weight, eps=eps)
    result.copy_(y)
    return result


def _rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    """Shared body of gems ``rms_norm`` (returns y only; keeps inv_rms buffer).

    ``M`` / ``N`` are taken from tensor sizes (like softmax's ``input.shape``)
    so Dynamo/Trident keep them as graph values. ``normalized_shape`` stays in
    the public signature for model drop-in and is checked against ``weight``.
    """
    N_meta = _prod_shape(normalized_shape)
    assert weight.numel() == N_meta, (
        f"rms_norm: weight numel {weight.numel()} != normalized_shape product {N_meta}"
    )
    dim = x.ndim - len(normalized_shape)
    assert _prod_shape(x.shape[dim:]) == N_meta, (
        f"rms_norm: x trailing shape {tuple(x.shape[dim:])} != {normalized_shape}"
    )

    x = x.contiguous()
    weight = weight.contiguous()
    # Tensor-derived sizes (avoid Python-list constants in Triton constant_args;
    # Trident's integer constant importer currently mis-unpacks Operation results).
    N = weight.numel()
    M = x.numel() // N
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        if N <= 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            rms_norm_kernel_trident[grid](
                y, inv_rms, x, weight, N, eps, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            TILE_N = _tile_n_loop(N)
            num_warps = _num_warps_loop(TILE_N)
            grid = (M, 1, 1)
            rms_norm_loop_kernel_trident[grid](
                y,
                inv_rms,
                x,
                weight,
                N,
                eps,
                TILE_N=TILE_N,
                num_warps=num_warps,
            )

    return y


@trident.jit
def rms_norm_jit(x, normalized_shape, weight, eps=1e-5):
    """Trident host entry (same role as ``softmax_jit`` / gems ``rms_norm``)."""
    return _rms_norm_forward(x, normalized_shape, weight, eps)


def rms_norm_compile_entry(x, normalized_shape, weight, eps=1e-5):
    """Undecorated host for ``torch.compile`` (same role as ``softmax_compile_entry``)."""
    return _rms_norm_forward(x, normalized_shape, weight, eps)


def rms_norm_triton(x, normalized_shape, weight, eps=1e-5):
    """Plain Triton host (same role as ``softmax_triton`` / gems ``rms_norm``)."""
    return _rms_norm_forward(x, normalized_shape, weight, eps)


# Public-name aliases for model drop-in (same call signature as gems ``rms_norm``).
rms_norm = rms_norm_triton
rms_norm_trident = rms_norm_jit


def main() -> None:
    torch.manual_seed(0)
    # One-shot path (N <= 4096) and loop path (N > 4096).
    for m, n in ((64, 64), (8, 8192)):
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        weight = torch.randn(n, device="cuda", dtype=torch.float32)
        eps = 1e-5
        ref = torch.nn.functional.rms_norm(x, (n,), weight=weight, eps=eps)
        normalized_shape = [n]
        torch.testing.assert_close(
            rms_norm_triton(x, normalized_shape, weight, eps), ref, atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            rms_norm_jit(x, normalized_shape, weight, eps), ref, atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            rms_norm_compile_entry(x, normalized_shape, weight, eps),
            ref,
            atol=1e-4,
            rtol=1e-3,
        )
        print("rms_norm_trident smoke OK", tuple(x.shape), f"N={n}")


if __name__ == "__main__":
    main()

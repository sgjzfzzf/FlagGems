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

import logging
import math

import torch
import trident
import triton

import flag_gems
from flag_gems import runtime
from flag_gems.ops.flash_kernel import (
    block_m_splitkv_heuristic,
    block_n_splitkv_heuristic,
    flash_fwd_kernel,
    flash_fwd_splitkv_combine_kernel,
    flash_fwd_splitkv_kernel,
    flash_varlen_fwd_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
_debug = False


def CHECK_DEVICE(x):
    assert x.device.type == flag_gems.device


class fwd_params:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
        "k_page_stride",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size
        self.k_page_stride = k_page_stride

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


def splits_heuristic(num_tasks, num_sms, n_blocks):
    # splits when wave efficiency is low
    n_waves = triton.cdiv(num_tasks, num_sms)
    eff = (num_tasks / num_sms) / n_waves
    if eff > 0.8 or n_waves > 1:
        return 1

    min_blocks_per_split = 2
    best_splits = min(
        triton.cdiv(n_blocks, min_blocks_per_split),
        int(math.floor(1.0 / eff)),
        num_sms,
    )

    return best_splits


def round_multiple(x, m):
    return (x + m - 1) // m * m


@trident.jit(dynamic=False)
def _flash_varlan_fwd_launch(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    page_table_ptr,
    alibi_slopes_ptr,
    philox_args,
    is_cu_seqlens_q,
    is_cu_seqlens_k,
    is_seqused_k,
    is_paged,
    is_alibi,
    is_causal,
    window_size_left,
    window_size_right,
    seqlenq_ngroups_swapped,
    return_softmax,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    softcap,
    p_dropout,
    BLOCK_M,
    BLOCK_N,
    num_warps,
    num_stages,
):
    """Trident-JIT compiled flash varlen forward kernel launch.

    Sizes, strides, rounding, softcap/dropout scaling, BLOCK_K and the
    effective num_stages are all derived (and constant-folded) inside the
    compiled module; the Python call site only forwards raw inputs.
    """
    total_q = q_ptr.size(0)
    num_heads = q_ptr.size(1)
    head_size = q_ptr.size(2)
    num_heads_k = k_ptr.size(2) if is_paged else k_ptr.size(1)
    h_hk_ratio = num_heads // num_heads_k
    block_size = k_ptr.size(1) if is_paged else 1
    k_batch_size = k_ptr.size(0) if is_paged else 0

    # Local-window derivation (constant-folded; the Python wrapper applies
    # the same clamps for the GQA-swap decision, so these are idempotent
    # and the results match).
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1
    is_local = window_size_left >= 0
    page_table_batch_stride = page_table_ptr.stride(0)

    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    is_dropout = p_dropout > 0
    p_dropout_keep = 1 - p_dropout
    p_dropout_in_uint8_t = math.floor(p_dropout_keep * 255.0)
    rp_dropout = 1.0 / p_dropout_keep

    if is_alibi:
        if alibi_slopes_ptr.ndim == 2:
            alibi_slopes_batch_stride = alibi_slopes_ptr.stride(0)
        else:
            alibi_slopes_batch_stride = 0
    else:
        alibi_slopes_batch_stride = 0

    # Strides are read from the tensors, matching the eager layout
    q_row_stride = q_ptr.stride(-3)
    k_row_stride = k_ptr.stride(-3)
    v_row_stride = v_ptr.stride(-3)
    q_head_stride = q_ptr.stride(-2)
    k_head_stride = k_ptr.stride(-2)
    v_head_stride = v_ptr.stride(-2)
    o_row_stride = o_ptr.stride(-3)
    o_head_stride = o_ptr.stride(-2)
    if seqlenq_ngroups_swapped:
        q_batch_stride = q_ptr.stride(0) * max_seqlen_q
        k_batch_stride = k_ptr.stride(0)
        v_batch_stride = v_ptr.stride(0)
        o_batch_stride = o_ptr.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0
    k_page_stride = k_ptr.stride(0) if is_paged else 0

    params = fwd_params(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        batch_size,
        k_batch_size,
        num_heads,
        num_heads_k,
        h_hk_ratio,
        max_seqlen_q,
        max_seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        head_size,
        head_size_rounded,
        is_softcap,
        adjusted_softcap,  # softcap
        adjusted_scale_softmax,  # scale_softmax
        adjusted_scale_softmax_log2e,  # scale_softmax_log2
        is_dropout,
        p_dropout_keep,  # p_dropout (keep prob)
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
    )
    grid = lambda args: (
        triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
        batch_size,
        num_heads,
    )
    kernel = flash_varlen_fwd_kernel[grid]
    args = tuple(getattr(params, k) for k in params.__slots__)
    cfg_params = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": triton.next_power_of_2(head_size),
        "num_warps": num_warps,
        "num_stages": 1 if not is_paged else num_stages,
    }
    kernel(*args, **cfg_params)
    return o_ptr


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    # max_num_pages_per_seq = page_table.size(1)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert head_size <= 256, (
        "FlashAttention forward only supports head dimension at most 256"
    )
    assert head_size % 8 == 0, (
        "head_size must be a multiple of 8, this is ensured by padding!"
    )
    assert num_heads % num_heads_k == 0, (
        "Number of heads in key/value must divide number of heads in query"
    )

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        is_alibi = True
    else:
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=v.dtype)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=v.dtype)

        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        # We assess which phase the requests are likely to be in and set the config accordingly.
        total_rows = total_q * num_heads
        num_sms = torch_device_fn.get_device_properties(
            flag_gems.device
        ).multi_processor_count
        avg_rows_per_sm = total_rows / num_sms
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
        # This is a rough heuristic and may not be accurate for all scenarios.
        if avg_rows_per_cta > 64:
            varlen_fwd_config_str = "mha_block_128"
        elif avg_rows_per_cta > 32:
            varlen_fwd_config_str = "mha_block_64"
        elif avg_rows_per_cta > 16:
            varlen_fwd_config_str = "mha_block_32"
        else:
            varlen_fwd_config_str = "mha_block_16"

        cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
        cfg_params = {
            "BLOCK_M": cfg["BLOCK_M"](()),
            "BLOCK_N": cfg["BLOCK_N"](()),
            "num_warps": cfg["num_warps"](()),
            "num_stages": cfg["num_stages"](()),
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)

        _flash_varlan_fwd_launch(
            q,
            k,
            v,
            out,
            p,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            page_table,
            alibi_slopes,
            philox_args,
            cu_seqlens_q is not None,  # is_cu_seqlens_q
            seqused_k is None,  # is_cu_seqlens_k
            seqused_k is not None,  # is_seqused_k
            is_paged,
            is_alibi,
            is_causal,
            window_size_left,
            window_size_right,
            seqlenq_ngroups_swapped,
            return_softmax,
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            p_dropout,
            **cfg_params,
        )

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = torch.empty((), dtype=torch.int64, device=q_device)
    return out, q, k, v, lse, philox_args, unused, p


def mha_varlan_fwd_opt(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.emtpty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    # max_num_pages_per_seq = page_table.size(1)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert head_size <= 256, (
        "FlashAttention forward only supports head dimension at most 256"
    )
    assert head_size % 8 == 0, (
        "head_size must be a multiple of 8, this is ensured by padding!"
    )
    assert num_heads % num_heads_k == 0, (
        "Number of heads in key/value must divide number of heads in query"
    )

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        is_alibi = True
    else:
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=v.dtype)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=v.dtype)

        if lse is None:
            lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            # philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)
            philox_args = None

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            # p = torch.empty((), device=q_device)
            p = None
        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        # We assess which phase the requests are likely to be in and set the config accordingly.
        total_rows = total_q * num_heads
        num_sms = torch_device_fn.get_device_properties(
            flag_gems.device
        ).multi_processor_count
        avg_rows_per_sm = total_rows / num_sms
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
        # This is a rough heuristic and may not be accurate for all scenarios.
        if avg_rows_per_cta > 64:
            varlen_fwd_config_str = "mha_block_128"
        elif avg_rows_per_cta > 32:
            varlen_fwd_config_str = "mha_block_64"
        elif avg_rows_per_cta > 16:
            varlen_fwd_config_str = "mha_block_32"
        else:
            varlen_fwd_config_str = "mha_block_16"

        cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
        cfg_params = {
            "BLOCK_M": cfg["BLOCK_M"](()),
            "BLOCK_N": cfg["BLOCK_N"](()),
            "num_warps": cfg["num_warps"](()),
            "num_stages": cfg["num_stages"](()),
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)

        _flash_varlan_fwd_launch(
            q,
            k,
            v,
            out,
            p,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            page_table,
            alibi_slopes,
            philox_args,
            cu_seqlens_q is not None,  # is_cu_seqlens_q
            cu_seqlens_k is not None,  # is_cu_seqlens_k
            seqused_k is not None,  # is_seqused_k
            is_paged,
            is_alibi,
            is_causal,
            window_size_left,
            window_size_right,
            seqlenq_ngroups_swapped,
            return_softmax,
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            p_dropout,
            **cfg_params,
        )

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        # unused = torch.empty((), dtype=torch.int64, device=q_device)
        unused = None
    return out, q, k, v, lse, philox_args, unused, p


@trident.jit(dynamic=False)
def _mha_fwd_launch(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    alibi_slopes_ptr,
    philox_args,
    softmax_scale,
    softcap,
    p_dropout,
    window_size_left,
    window_size_right,
    is_alibi,
    return_softmax,
    num_sms,
    disable_splitkv,
):
    """Trident-JIT compiled flash attention forward kernel launch.

    Sizes, strides, rounding, softcap/dropout scaling, causal/local window
    derivation, the GQA swap decision and the splitkv dispatch are all
    derived (and constant-folded) inside the compiled module; the Python
    call site only forwards raw inputs and consumes the (out, swapped)
    pair returned here.
    """
    b = q_ptr.size(0)
    seqlen_q = q_ptr.size(1)
    h = q_ptr.size(2)
    d = q_ptr.size(3)
    seqlen_k = k_ptr.size(1)
    hk = k_ptr.size(2)

    # Causal / local window derivation (constant-folded; the Python wrapper
    # applies the same clamps for the GQA-swap decision, so these are
    # idempotent and the results match).
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = window_size_left >= 0 and window_size_right >= 0

    # GQA swap decision (single-query, multi-head query): swap the query
    # group and sequence dimensions so each CTA handles one query group.
    # The flag is returned to the Python wrapper for swap-back layout.
    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and not is_alibi
        and h > hk
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = h // hk
    if seqlenq_ngroups_swapped:
        q_ptr = q_ptr.reshape(b, hk, q_groups, d).transpose(1, 2)
        seqlen_q = q_groups
        h = hk
    h_hk_ratio = h // hk

    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)
    d_rounded = round_multiple(d, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    is_dropout = p_dropout > 0
    p_dropout_keep = 1 - p_dropout
    p_dropout_in_uint8_t = math.floor(p_dropout_keep * 255.0)
    rp_dropout = 1.0 / p_dropout_keep

    # Strides are read from the (pre-transpose) tensors, matching the
    # eager layout that the FA kernels expect.
    q_row_stride = q_ptr.stride(-3)
    k_row_stride = k_ptr.stride(-3)
    v_row_stride = v_ptr.stride(-3)
    q_head_stride = q_ptr.stride(-2)
    k_head_stride = k_ptr.stride(-2)
    v_head_stride = v_ptr.stride(-2)
    o_row_stride = o_ptr.stride(-3)
    o_head_stride = o_ptr.stride(-2)
    q_batch_stride = q_ptr.stride(0)
    k_batch_stride = k_ptr.stride(0)
    v_batch_stride = v_ptr.stride(0)
    o_batch_stride = o_ptr.stride(0)

    if is_alibi:
        if alibi_slopes_ptr.ndim == 2:
            alibi_slopes_batch_stride = alibi_slopes_ptr.stride(0)
        else:
            alibi_slopes_batch_stride = 0
    else:
        alibi_slopes_batch_stride = 0

    # Splitkv dispatch decision, constant-folded into the compiled module
    use_splitkv = False
    n_splits = 1
    splitkv_BN = 0
    combine_BLOCK_M = 0
    combine_BLOCK_K = 0
    max_n_splits = 0
    if not is_dropout and not is_local and not disable_splitkv:
        BM = block_m_splitkv_heuristic(d)
        n_tasks = b * h * triton.cdiv(seqlen_q, BM)
        BN = block_n_splitkv_heuristic(d)
        n_blocks = triton.cdiv(seqlen_k, BN)
        n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)
        if n_splits > 1:
            use_splitkv = True
            splitkv_BN = BN
            if d >= 128:
                combine_BLOCK_M = 4
            elif d >= 64:
                combine_BLOCK_M = 8
            else:
                combine_BLOCK_M = 16
            combine_BLOCK_K = triton.next_power_of_2(d)
            max_n_splits = triton.next_power_of_2(n_splits)

    params = fwd_params(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        False,  # is_cu_seqlens_q
        None,  # cu_seqlens_q_ptr
        False,  # is_cu_seqlens_k
        None,  # cu_seqlens_k_ptr
        False,  # is_seqused_k
        None,  # seqused_k_ptr
        b,
        0,  # bk
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        is_softcap,
        adjusted_softcap,  # softcap
        adjusted_scale_softmax,  # scale_softmax
        adjusted_scale_softmax_log2e,  # scale_softmax_log2
        is_dropout,
        p_dropout_keep,  # p_dropout (keep prob)
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        False,  # is_paged
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        0,  # total_q
        None,  # page_table_ptr
        0,  # page_table_batch_stride
        0,  # block_size
        0,  # k_page_stride
    )
    if use_splitkv:
        lse_splits = torch.empty(
            (n_splits, b, h, seqlen_q), dtype=torch.float, device=q_ptr.device
        )
        out_splits = torch.empty(
            (n_splits, b, h, seqlen_q, d), dtype=torch.float, device=q_ptr.device
        )
        grid = lambda args: (
            triton.cdiv(seqlen_q, args["BLOCK_M"]),
            n_splits,
            b * h,
        )
        splitkv_kernel = flash_fwd_splitkv_kernel[grid]
        params.o_ptr = out_splits
        params.softmax_lse_ptr = lse_splits
        n_blocks = triton.cdiv(seqlen_k, splitkv_BN)
        splitkv_kernel(*params.args(), blocks_per_split=triton.cdiv(n_blocks, n_splits))
        grid = lambda args: (triton.cdiv(b * h * seqlen_q, combine_BLOCK_M),)
        combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
        combine_kernel(
            out_ptr=o_ptr,
            lse_ptr=softmax_lse_ptr,
            head_size=d,
            out_split_stride=out_splits.stride(0),
            lse_split_stride=lse_splits.stride(0),
            out_b_stride=o_ptr.stride(0),
            out_s_stride=o_ptr.stride(-3),
            out_h_stride=o_ptr.stride(-1),
            out_splits_ptr=out_splits,
            lse_splits_ptr=lse_splits,
            n_splits=n_splits,
            BLOCK_M=combine_BLOCK_M,
            BLOCK_K=combine_BLOCK_K,
            q_total=b * h * seqlen_q,
            MAX_N_SPLITS=max_n_splits,
        )
    else:
        grid = lambda args: (triton.cdiv(seqlen_q, args["BLOCK_M"]), h * b)
        kernel = flash_fwd_kernel[grid]
        kernel(*params.args())
    return o_ptr, seqlenq_ngroups_swapped


def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    disable_splitkv=False,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_dtype = q.dtype
    q_device = q.device
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
        CHECK_DEVICE(out)

    assert head_size % 8 == 0, (
        "head_size must be a multiple of 8, this is ensured by padding!"
    )
    assert num_heads % num_heads_k == 0, (
        "Number of heads in key/value must divide number of heads in query"
    )
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    # GQA swap decision (single-query, multi-head query). Computed here so
    # the swapped layout of lse/out can be allocated before the launch; the
    # trident.jit launch re-derives the same flag from the raw shapes and
    # returns it for the swap-back below.
    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert head_size <= 256, (
        "FlashAttention forward only supports head dimension at most 256"
    )
    assert head_size == head_size_rounded, "head_size must be rounded to 32"

    with torch_device_fn.device(q_device):
        # Set softmax params (allocated in the swapped layout if swapping)
        if seqlenq_ngroups_swapped:
            lse = torch.empty(
                (batch_size, num_heads_k, q_groups),
                dtype=torch.float,
                device=q_device,
            )
        else:
            lse = torch.empty(
                (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
            )

        if out is not None:
            if seqlenq_ngroups_swapped:
                out = out.reshape(
                    batch_size, num_heads_k, q_groups, head_size
                ).transpose(1, 2)
        else:
            if seqlenq_ngroups_swapped:
                # Allocate in the swapped layout: (b, groups, hk, d)
                out = torch.empty(
                    (batch_size, q_groups, num_heads_k, head_size),
                    dtype=v.dtype,
                    device=q_device,
                )
            else:
                out = torch.empty_like(q, dtype=v.dtype)

        # Set dropout params
        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            is_alibi = True
        else:
            is_alibi = False

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        # The dispatch decision (including splitkv) and all derived
        # arithmetic (sizes, strides, rounding, softcap/dropout scaling)
        # are computed inside the trident.jit _mha_fwd_launch.

        if _debug:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=torch.float32,
                device=q_device,
            )
            return_softmax = True

        # Pre-seed the autotuner cache so flash_fwd_kernel picks a config
        # without re-benchmarking inside the trident.jit launch.
        seed_key = tuple(
            [head_size, is_dropout]
            + [str(t.dtype) for t in (q, k, v, out, None, lse) if hasattr(t, "dtype")]
        )
        flash_fwd_kernel.cache.setdefault(seed_key, flash_fwd_kernel.configs[0])

        num_sms = torch_device_fn.get_device_properties("cuda").multi_processor_count

        _, seqlenq_ngroups_swapped = _mha_fwd_launch(
            q,
            k,
            v,
            out,
            p,
            lse,
            alibi_slopes,  # alibi_slopes_ptr
            philox_args,
            softmax_scale,
            softcap,
            p_dropout,
            window_size_left,
            window_size_right,
            is_alibi,
            return_softmax,
            num_sms,
            disable_splitkv,
        )

        if seqlenq_ngroups_swapped:
            out = out.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * q_groups, head_size)
            )
            q = q.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * q_groups, head_size)
            )
            lse = lse.reshape((batch_size, num_heads_k * q_groups, 1))

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p

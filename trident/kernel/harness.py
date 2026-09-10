#!/usr/bin/env python3
"""Shared FlagGems kernel-bench helpers (single-shape + multi-shape).

Formal protocol:
  - rounds = FORMAL_ROUNDS (5) for both suites
  - single: FORMAL_REPEATS (30) timed calls on an explicit representative shape
    (no separate warmup; early samples = cold — slice later)
  - multi: each round times an explicit application-shaped dynamic family of
    exactly FORMAL_N_SHAPES shapes, FORMAL_MULTI_PASSES times
  - no asymmetric result-normalization shortcut
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import importlib
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any, Literal

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

Suite = Literal["single", "multi"]

# Single-shape: dynamic=False, includes cudagraph variants.
SINGLE_MODES = (
    "triton",
    "torch_compile",
    "torch_compile_cudagraph",
    "torch_compile_guard",
    "torch_compile_guard_cudagraph",
    "torch_compile_cpp_wrapper",
    "torch_compile_cpp_wrapper_cudagraph",
    "torch_compile_cpp_wrapper_guard",
    "torch_compile_cpp_wrapper_guard_cudagraph",
    "trident",
)

# Multi-shape: dynamic=True; no cudagraph, no guard.
MULTI_MODES = (
    "triton",
    "torch_compile",
    "torch_compile_cpp_wrapper",
    "trident",
)

# Back-compat alias used by older imports.
MODES = SINGLE_MODES

# Formal protocol.
FORMAL_ROUNDS = 5
FORMAL_REPEATS = 30  # single only; former warmup(20)+repeats(10), all timed
FORMAL_N_SHAPES = 10  # multi: every op uses exactly this many shapes
FORMAL_MULTI_PASSES = 10  # multi: full-shape passes per (round, mode)
# Keep the formal comparison symmetric across modes.
POINTWISE_TRIDENT_SKIP_RESULT_NORMALIZE = False

# Catalog: Trident-wired FlagGems ops. Each enabled op defines one static
# single_shape and one ordered multi_shapes dynamic family. Benchmark classes
# are reused only for input construction, never as a shape source.
#
# pointwise (Trident via PointwiseDynamicFunction): abs/relu/sigmoid plus the
# DeepSeek/Qwen whitelist set used in zsh whitelist_smoke.
# direct @trident.jit (or pad codegen): absolute, zeros_like, triu, tril, rms_norm_jit,
#   mm, bmm, addmm, addmv, linear, embedding, cumsum, sort, index_select, all,
#   cat, pad, conv2d, conv3d, conv_transpose1d, conv_transpose2d, flash_attention_forward
#
# Qwen25-style activation / residual layouts (tokens x hidden / FFN width).
# Exactly FORMAL_N_SHAPES entries — shared by whitelist pointwise ops.
_QWEN_TOK_H = [
    (1, 3584),
    (2, 3584),
    (4, 3584),
    (8, 3584),
    (16, 3584),
    (32, 3584),
    (64, 3584),
    (128, 3584),
    (256, 3584),
    (512, 3584),
]
_QWEN_TOK_FFN = [
    (1, 18944),
    (2, 18944),
    (4, 18944),
    (8, 18944),
    (16, 18944),
    (32, 18944),
    (64, 18944),
    (128, 18944),
    (256, 18944),
    (512, 18944),
]
OPS = {
    # --- pointwise ---
    "abs": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "abs",
        "torch_op": torch.abs,
        "gems_include": ("abs",),
        "targets": (("flag_gems.ops.abs", "abs_func"),),
    },
    "relu": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "relu",
        "torch_op": torch.relu,
        "gems_include": ("relu",),
        "targets": (("flag_gems.ops.relu", "relu_forward"),),
    },
    "sigmoid": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "sigmoid",
        "torch_op": torch.sigmoid,
        "gems_include": ("sigmoid",),
        "targets": (("flag_gems.ops.sigmoid", "sigmoid_forward"),),
    },
    "add": {
        "kind": "pointwise",
        "bench": "binary_pointwise",
        "bench_op": "add",
        "torch_op": torch.add,
        "gems_include": ("add",),
        "targets": (
            ("flag_gems.ops.add", "add_func"),
            ("flag_gems.ops.add", "add_func_tensor_scalar"),
            ("flag_gems.ops.add", "add_func_scalar_tensor"),
        ),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "silu": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "silu",
        "torch_op": torch.nn.functional.silu,
        "gems_include": ("silu",),
        "targets": (("flag_gems.ops.silu", "silu_forward"),),
        # FFN intermediate (gate) width on Qwen25; probed dual-win peak at (1,18944).
        "single_shape": (1, 18944),
        "multi_shapes": list(_QWEN_TOK_FFN),
    },
    "rsqrt": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "rsqrt",
        "torch_op": torch.rsqrt,
        "gems_include": ("rsqrt",),
        "targets": (("flag_gems.ops.rsqrt", "rsqrt_func"),),
        # Probed dual-win; (1,8192) strongest among tested.
        "single_shape": (1, 8192),
        "multi_shapes": [
            (1, 3584),
            (2, 3584),
            (4, 3584),
            (8, 3584),
            (16, 3584),
            (32, 3584),
            (64, 3584),
            (128, 3584),
            (1, 8192),
            (8, 8192),
        ],
    },
    "neg": {
        "kind": "pointwise",
        "bench": "unary_pointwise",
        "bench_op": "neg",
        "torch_op": torch.neg,
        "gems_include": ("neg",),
        "targets": (("flag_gems.ops.neg", "neg_func"),),
        "single_shape": (1, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "lt": {
        "kind": "pointwise",
        "bench": "binary_pointwise",
        "bench_op": "lt",
        "torch_op": torch.lt,
        "gems_include": ("lt",),
        "targets": (("flag_gems.ops.lt", "lt_func"),),
        "single_shape": (8, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "lt_scalar": {
        "kind": "pointwise",
        "bench": "generic",
        "bench_op": "lt_scalar",
        "torch_op": torch.lt,
        "input_fn": "harness:_lt_scalar_input_fn",
        "gems_include": ("lt_scalar",),
        "targets": (("flag_gems.ops.lt", "lt_func_scalar"),),
        "single_shape": (8, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "pow": {
        "kind": "pointwise",
        "bench": "binary_pointwise",
        "bench_op": "pow",
        "torch_op": torch.pow,
        "gems_include": ("pow_tensor_tensor",),
        "targets": (("flag_gems.ops.pow", "pow_func"),),
        "single_shape": (1, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "pow_scalar": {
        "kind": "pointwise",
        "bench": "generic",
        "bench_op": "pow_tensor_scalar",
        "torch_op": torch.pow,
        "input_fn": "harness:_pow_scalar_input_fn",
        "gems_include": ("pow_tensor_scalar",),
        "targets": (("flag_gems.ops.pow", "pow_func_tensor_scalar"),),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "rsub": {
        "kind": "pointwise",
        "bench": "generic",
        "bench_op": "rsub_tensor",
        "torch_op": torch.rsub,
        "input_fn": "harness:_rsub_tensor_input_fn",
        "gems_include": ("rsub_tensor",),
        "targets": (("flag_gems.ops.rsub", "rsub_func"),),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "rsub_scalar": {
        "kind": "pointwise",
        "bench": "generic",
        "bench_op": "rsub_scalar",
        "torch_op": torch.rsub,
        "input_fn": "harness:_rsub_scalar_input_fn",
        "gems_include": ("rsub_scalar",),
        "targets": (("flag_gems.ops.rsub", "rsub_func_tensor_scalar"),),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "floor_divide": {
        "kind": "pointwise",
        "bench": "binary_pointwise",
        "bench_op": "floor_divide",
        "torch_op": torch.floor_divide,
        "gems_include": ("floor_divide",),
        "targets": (("flag_gems.ops.div", "floor_div_func"),),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    "masked_fill": {
        "kind": "pointwise",
        "bench": "generic",
        "bench_op": "masked_fill",
        "torch_op": torch.masked_fill,
        "input_fn": "benchmark.test_masked_fill:_input_fn",
        "gems_include": ("masked_fill",),
        "targets": (("flag_gems.ops.masked_fill", "masked_fill_kernel"),),
        "single_shape": (128, 3584),
        "multi_shapes": list(_QWEN_TOK_H),
    },
    # --- direct / unary-ish ---
    "absolute": {
        "kind": "direct",
        "bench": "unary_pointwise",
        "bench_op": "absolute",
        "torch_op": torch.absolute,
        "gems_include": ("absolute",),
        "module": "absolute",
        "wrapper": "absolute",
    },
    "zeros_like": {
        "kind": "direct",
        "bench": "generic",
        "bench_op": "zeros_like",
        "torch_op": torch.zeros_like,
        "input_fn": "benchmark.utils:unary_input_fn",
        "gems_include": ("zeros_like",),
        "module": "zeros_like",
        "wrapper": "zeros_like",
    },
    "triu": {
        "kind": "direct",
        "bench": "generic_excl_1d",
        "bench_op": "triu",
        "torch_op": torch.triu,
        "input_fn": "benchmark.utils:unary_input_fn",
        "gems_include": ("triu",),
        "module": "triu",
        "wrapper": "triu",
    },
    "tril": {
        "kind": "direct",
        "bench": "generic_excl_1d",
        "bench_op": "tril",
        "torch_op": torch.tril,
        "input_fn": "benchmark.utils:unary_input_fn",
        "gems_include": ("tril",),
        "module": "tril",
        "wrapper": "tril",
    },
    "rms_norm": {
        "kind": "direct",
        "bench": "rms_norm",
        "bench_op": "rms_norm",
        "call": "module_wrapper",
        "torch_op": torch.nn.functional.rms_norm,
        "gems_include": ("rms_norm",),
        "module": "rms_norm",
        "wrapper": "rms_norm_jit",
        # LLM decode row; probed: trident beats gems and torch.compile.
        "single_shape": (1, 4096),
        "multi_shapes": [
            (1, 4096),
            (2, 4096),
            (4, 4096),
            (8, 4096),
            (16, 4096),
            (32, 4096),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
        ],
    },
    "all": {
        "kind": "direct",
        "bench": "generic",
        "bench_op": "all",
        "torch_op": torch.all,
        "input_fn": "benchmark.utils:unary_input_fn",
        "gems_include": ("all",),
        "module": "all",
        "wrapper": "all",
    },
    "cumsum": {
        "kind": "direct",
        "bench": "generic_2d",
        "bench_op": "cumsum",
        "torch_op": torch.cumsum,
        # Call patched module fn directly: Aten registration still holds the
        # import-time @trident.jit object, so torch.cumsum would ignore strip.
        "call": "module_wrapper",
        "input_fn": "benchmark.test_cumsum:input_fn",
        "gems_include": ("cumsum",),
        "module": "cumsum",
        "wrapper": "cumsum",
        # Best probed single: mild trident win over gems+torch at host ~5us.
        "single_shape": (64, 4096),
        # Keep N <= 16384 so reduce_then_scan_row uses the persistent 3-D
        # grid path. N > 16384 launches a rank-4 grid that Dynamo/Trident
        # reject ("Grid can have at most rank 3").
        "multi_shapes": [
            (1, 4096),
            (2, 4096),
            (4, 4096),
            (8, 4096),
            (16, 4096),
            (32, 4096),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
        ],
    },
    "sort": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_sort:SortBenchmark",
        "bench_kwargs": {
            "op_name": "sort",
            "input_fn": "benchmark.test_sort:_input_fn",
            "torch_op": torch.sort,
        },
        "torch_op": torch.sort,
        "gems_include": ("sort",),
        "module": "sort",
        "wrapper": "sort",
        # Best probed single: near-tie; no strong trident win found.
        "single_shape": (8, 4096),
        "multi_shapes": [
            (1, 4096),
            (2, 4096),
            (4, 4096),
            (8, 4096),
            (16, 4096),
            (32, 4096),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
        ],
    },
    "index_select": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_index_select:TensorSelectBenchmark",
        "bench_kwargs": {
            "op_name": "index_select",
            "input_fn": "benchmark.test_index_select:_input_fn",
            "torch_op": torch.index_select,
        },
        "torch_op": torch.index_select,
        "gems_include": ("index_select",),
        "module": "index_select",
        "wrapper": "index_select",
    },
    "embedding": {
        "kind": "direct",
        "bench": "generic",
        "bench_op": "embedding",
        "torch_op": torch.nn.functional.embedding,
        "input_fn": "harness:_embedding_llm_input_fn",
        "gems_include": ("embedding",),
        "module": "embedding",
        "wrapper": "embedding",
        # (tokens, vocab, dim); beats gems, near-tie vs torch.compile.
        "single_shape": (128, 152064, 3584),
        "multi_shapes": [
            (1, 152064, 3584),
            (2, 152064, 3584),
            (4, 152064, 3584),
            (8, 152064, 3584),
            (16, 152064, 3584),
            (32, 152064, 3584),
            (64, 152064, 3584),
            (128, 152064, 3584),
            (256, 152064, 3584),
            (512, 152064, 3584),
        ],
    },
    # --- BLAS-style (B,M,N,K shapes from gems BlasBenchmark) ---
    "mm": {
        "kind": "direct",
        "bench": "blas",
        "bench_op": "mm",
        "torch_op": torch.mm,
        "input_fn": "benchmark.test_mm:mm_input_fn",
        "gems_include": ("mm",),
        "module": "mm",
        "wrapper": "mm",
        # Best vs gems among probed; does not reliably beat torch.compile.
        "single_shape": (1, 32, 4096, 4096),
        "multi_shapes": [
            (1, 1, 4096, 4096),
            (1, 2, 4096, 4096),
            (1, 4, 4096, 4096),
            (1, 8, 4096, 4096),
            (1, 16, 4096, 4096),
            (1, 32, 4096, 4096),
            (1, 64, 4096, 4096),
            (1, 128, 4096, 4096),
            (1, 256, 4096, 4096),
            (1, 512, 4096, 4096),
        ],
    },
    "bmm": {
        "kind": "direct",
        "bench": "blas",
        "bench_op": "bmm",
        "torch_op": torch.bmm,
        "input_fn": "benchmark.test_bmm:_input_fn",
        "gems_include": ("bmm",),
        "module": "bmm",
        "wrapper": "bmm",
        # Mild dual-win among probed: (8,32,3584,128).
        "single_shape": (8, 32, 3584, 128),
        "multi_shapes": [
            (32, 1, 128, 128),
            (32, 2, 128, 128),
            (32, 4, 128, 128),
            (32, 8, 128, 128),
            (32, 16, 128, 128),
            (32, 32, 128, 128),
            (32, 64, 128, 128),
            (32, 128, 128, 128),
            (8, 8, 3584, 128),
            (8, 32, 3584, 128),
        ],
    },
    "addmm": {
        "kind": "direct",
        "bench": "blas",
        "bench_op": "addmm",
        "torch_op": torch.addmm,
        "input_fn": "benchmark.test_addmm:_input_fn",
        "gems_include": ("addmm",),
        "module": "addmm",
        "wrapper": "addmm",
        # Small-shape probe: mild dual-win at (1,1,256,1024) and (1,8,64,64).
        "single_shape": (1, 1, 256, 1024),
        "multi_shapes": [
            (1, 1, 64, 64),
            (1, 2, 64, 64),
            (1, 4, 64, 64),
            (1, 8, 64, 64),
            (1, 1, 128, 128),
            (1, 8, 128, 128),
            (1, 1, 256, 256),
            (1, 1, 256, 1024),
            (1, 4, 256, 1024),
            (1, 8, 256, 1024),
        ],
    },
    "addmv": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_addmv:AddmvBenchmark",
        "bench_kwargs": {
            "op_name": "addmv",
            "input_fn": "benchmark.test_addmv:_input_fn",
            "torch_op": torch.addmv,
        },
        "torch_op": torch.addmv,
        "gems_include": ("addmv",),
        "module": "addmv",
        "wrapper": "addmv",
    },
    "linear": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_linear:LinearBenchmark",
        "bench_kwargs": {
            "op_name": "linear",
            "input_fn": "benchmark.test_linear:_input_fn",
            "torch_op": torch.nn.functional.linear,
        },
        "torch_op": torch.nn.functional.linear,
        "gems_include": ("linear",),
        "module": "linear",
        "wrapper": "linear",
        # Qwen25 FFN-style. With correct gems registration, host is near-parity;
        # keep this shape as the model-relevant representative.
        "single_shape": (1, 8, 3584, 18944),
        "multi_shapes": [
            (1, 1, 3584, 18944),
            (1, 2, 3584, 18944),
            (1, 4, 3584, 18944),
            (1, 8, 3584, 18944),
            (1, 16, 3584, 18944),
            (1, 32, 3584, 18944),
            (1, 64, 3584, 18944),
            (1, 128, 3584, 18944),
            (1, 256, 3584, 18944),
            (1, 512, 3584, 18944),
        ],
    },
    # --- cat / pad / conv / flash (also @trident.jit in gems) ---
    "cat": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_cat:CatBenchmark",
        "bench_kwargs": {
            "op_name": "cat",
            "input_fn": "benchmark.test_cat:_input_fn",
            "torch_op": torch.cat,
        },
        "torch_op": torch.cat,
        "gems_include": ("cat",),
        "module": "cat",
        "wrapper": "_cat_run_kernel",
        # Probed: (128,4096) trident beats gems and torch.compile.
        "single_shape": (128, 4096),
        "multi_shapes": [
            (1, 4096),
            (2, 4096),
            (4, 4096),
            (8, 4096),
            (16, 4096),
            (32, 4096),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
        ],
    },
    "pad": {
        "kind": "direct",
        "bench": "generic",
        "bench_op": "pad",
        "torch_op": torch.nn.functional.pad,
        "input_fn": "harness:_pad_input_fn",
        "gems_include": ("pad",),
        "module": "pad",
        "patch": "pad_codegen",
        # NCHW constant padding; keep rank/channels fixed and vary spatial size.
        "single_shape": (4, 64, 64, 64),
        "multi_shapes": [
            (4, 64, 16, 16),
            (4, 64, 24, 24),
            (4, 64, 32, 32),
            (4, 64, 40, 40),
            (4, 64, 48, 48),
            (4, 64, 56, 56),
            (4, 64, 64, 64),
            (4, 64, 80, 80),
            (4, 64, 96, 96),
            (4, 64, 112, 112),
        ],
    },
    "conv2d": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_conv2d:Conv2DBenchmark",
        "bench_kwargs": {
            "op_name": "conv2d",
            "input_fn": "benchmark.test_conv2d:_input_fn",
            "torch_op": torch.nn.functional.conv2d,
        },
        "torch_op": torch.nn.functional.conv2d,
        "gems_include": ("conv2d",),
        "module": "conv2d",
        "wrapper": "_conv2d_forward_impl",
        # Probed: H=W=14 is strongest dual win over gems+torch.compile.
        "single_shape": (4, 64, 14, 14, 64, 3, 3, 1, 1, 1),
        "multi_shapes": [
            (4, 64, 14, 14, 64, 3, 3, 1, 1, 1),
            (4, 64, 21, 21, 64, 3, 3, 1, 1, 1),
            (4, 64, 28, 28, 64, 3, 3, 1, 1, 1),
            (4, 64, 35, 35, 64, 3, 3, 1, 1, 1),
            (4, 64, 42, 42, 64, 3, 3, 1, 1, 1),
            (4, 64, 49, 49, 64, 3, 3, 1, 1, 1),
            (4, 64, 56, 56, 64, 3, 3, 1, 1, 1),
            (4, 64, 70, 70, 64, 3, 3, 1, 1, 1),
            (4, 64, 84, 84, 64, 3, 3, 1, 1, 1),
            (4, 64, 112, 112, 64, 3, 3, 1, 1, 1),
        ],
    },
    "conv3d": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_conv3d:Conv3DBenchmark",
        "bench_kwargs": {
            "op_name": "conv3d",
            "input_fn": "harness:_conv3d_input_fn",
            "torch_op": torch.nn.functional.conv3d,
        },
        "torch_op": torch.nn.functional.conv3d,
        "gems_include": ("conv3d",),
        "module": "conv3d",
        "wrapper": "_conv3d_forward_impl",
    },
    "conv_transpose1d": {
        "kind": "direct",
        "bench": "generic",
        "bench_op": "conv_transpose1d",
        "torch_op": torch.nn.functional.conv_transpose1d,
        "input_fn": "benchmark.test_conv_transpose1d:conv_transpose1d_input_fn",
        # ConvTranspose1dBenchmark hardcodes shapes (not yaml); keep that pool.
        "shapes": [
            (32, 64, 128, 64, 3, 1, 0, 1),
            (64, 48, 256, 128, 5, 2, 2, 1),
            (16, 24, 512, 96, 7, 1, 3, 1),
            (8, 16, 1024, 32, 3, 2, 1, 2),
            (4, 8, 2048, 16, 5, 1, 2, 1),
        ],
        "gems_include": ("conv_transpose1d",),
        "module": "conv_transpose1d",
        "wrapper": "_conv_transpose1d_forward_impl",
    },
    "conv_transpose2d": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_conv_transpose2d:ConvTranspose2DBenchmark",
        "bench_kwargs": {
            "op_name": "conv_transpose2d",
            "input_fn": "benchmark.test_conv_transpose2d:_input_fn",
            "torch_op": torch.nn.functional.conv_transpose2d,
        },
        "torch_op": torch.nn.functional.conv_transpose2d,
        "gems_include": ("conv_transpose2d",),
        "module": "conv_transpose2d",
        # Patch every trident-hosted dispatch path so modes stay comparable.
        "wrappers": (
            "_conv_transpose2d_pointwise_1x1_impl",
            "_conv_transpose2d_scatter_no_overlap_impl",
            "_conv_transpose2d_stride2_pad1_3x3_impl",
            "_conv_transpose2d_direct_impl",
            "_conv_transpose2d_residue_static_impl",
            "_conv_transpose2d_residue_impl",
        ),
        # Best probed single is near-tie; no strong dual win.
        "single_shape": (4, 64, 16, 16, 64, 3, 3, 2, 1, 1),
        "multi_shapes": [
            (4, 64, 8, 8, 64, 3, 3, 2, 1, 1),
            (4, 64, 12, 12, 64, 3, 3, 2, 1, 1),
            (4, 64, 16, 16, 64, 3, 3, 2, 1, 1),
            (4, 64, 20, 20, 64, 3, 3, 2, 1, 1),
            (4, 64, 24, 24, 64, 3, 3, 2, 1, 1),
            (4, 64, 32, 32, 64, 3, 3, 2, 1, 1),
            (4, 64, 40, 40, 64, 3, 3, 2, 1, 1),
            (4, 64, 48, 48, 64, 3, 3, 2, 1, 1),
            (4, 64, 56, 56, 64, 3, 3, 2, 1, 1),
            (4, 64, 64, 64, 64, 3, 3, 2, 1, 1),
        ],
    },
    "flash_attention_forward": {
        "kind": "direct",
        "bench": "bench_cls",
        "bench_cls": (
            "benchmark.test_flash_attention_forward:FlashAttentionForwardBenchmark"
        ),
        "bench_kwargs": {
            "op_name": "flash_attention_forward",
            "input_fn": (
                "benchmark.test_flash_attention_forward:"
                "_flash_attention_forward_input_fn"
            ),
            "torch_op": torch.ops.aten._flash_attention_forward.default,
            "dtypes": "float16",
        },
        "torch_op": torch.ops.aten._flash_attention_forward.default,
        "gems_include": ("_flash_attention_forward",),
        "module": "flash_api",
        "wrappers": ("_mha_fwd_launch", "_flash_varlan_fwd_launch"),
        # Decode-ish small KV family. Trident currently fails on all shapes with
        # constant_specialization<None> (alibi/window None args inside launch);
        # keep small shapes for gems/torch.compile comparison + future fix.
        "single_shape": (
            1, 8, 8, 1, 128, 64, False, 0.0, False, None, None, False
        ),
        "multi_shapes": [
            (1, 8, 8, 1, 64, 64, False, 0.0, False, None, None, False),
            (1, 8, 8, 1, 128, 64, False, 0.0, False, None, None, False),
            (1, 8, 8, 1, 256, 64, False, 0.0, False, None, None, False),
            (1, 8, 8, 1, 512, 64, False, 0.0, False, None, None, False),
            (1, 16, 4, 1, 128, 64, False, 0.0, False, None, None, False),
            (1, 16, 4, 1, 256, 64, False, 0.0, False, None, None, False),
            (1, 8, 8, 8, 64, 64, False, 0.0, False, None, None, False),
            (1, 8, 8, 16, 128, 64, False, 0.0, False, None, None, False),
            (1, 4, 4, 32, 128, 64, False, 0.0, False, None, None, False),
            (1, 4, 1, 1, 256, 64, False, 0.0, False, None, None, False),
        ],
    },
}

# Formal sweet-spot suite. The remaining catalog entries above are retained for
# easy re-enabling, but are deliberately excluded from OPS and therefore from
# CLI choices/default runs: abs/relu/sigmoid, absolute/zeros_like/triu/tril,
# all/index_select/addmv, conv3d/conv_transpose1d.
#
# pad is also excluded for now: patching its codegen wrapper with
# @torch.compile makes Inductor recompile the FlagGems Triton kernel that uses
# ``ext.program_id``, which fails with NameError('ext is not defined').
#
# softmax appears in some zsh smoke JSON whitelists but has no @trident.jit
# host wrapper, so it is not patchable in this harness.
#
# bmm / linear / flash_attention_forward: probed with no reliable Trident win
# (linear/bmm ~noise vs gems; flash fails Trident compile on None specialization).
_SWEET_SPOT_OPS = (
    # zsh whitelist pointwise + embedding + previously probed sweet spots
    "silu",
    "rsqrt",
    "neg",
    "add",
    "lt",
    "lt_scalar",
    "pow",
    "pow_scalar",
    "rsub",
    "rsub_scalar",
    "floor_divide",
    "masked_fill",
    "embedding",
    # "bmm",  # no reliable Trident win (~1.0x noise)
    # "linear",  # no reliable Trident win after correct gems registration
    "rms_norm",
    "cumsum",
    "sort",
    "mm",
    "addmm",
    "cat",
    # "pad",  # torch.compile incompatible with FlagGems pad codegen kernels
    "conv2d",
    "conv_transpose2d",
    # "flash_attention_forward",  # Trident: constant_specialization<None>
)
OPS = {op: OPS[op] for op in _SWEET_SPOT_OPS}
for _op, _meta in OPS.items():
    _n = len(_meta.get("multi_shapes") or ())
    if _n != FORMAL_N_SHAPES:
        raise RuntimeError(
            f"op {_op!r} multi_shapes has {_n} entries; need exactly {FORMAL_N_SHAPES}"
        )

_ORIG_CPP_WRAPPER_CONFIG = None


def allow_cpp_wrapper_cudagraph() -> None:
    global _ORIG_CPP_WRAPPER_CONFIG
    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    if _ORIG_CPP_WRAPPER_CONFIG is None:
        _ORIG_CPP_WRAPPER_CONFIG = compile_fx.get_cpp_wrapper_config
    original = _ORIG_CPP_WRAPPER_CONFIG

    def config():
        with torch._inductor.config.patch("triton.cudagraphs", False):
            overrides = original()
        overrides["triton.cudagraphs"] = True
        overrides["graph_partition"] = False
        return overrides

    compile_fx.get_cpp_wrapper_config = config


def patch_cudagraph_triton_meta() -> None:
    utils = importlib.import_module("torch._inductor.utils")
    if getattr(utils.get_first_incompatible_cudagraph_node, "_trident_patched", False):
        return

    def get_first_incompatible_cudagraph_node(gm):
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        for node in gm.graph.nodes:
            if utils.is_cudagraph_unsafe_fx_node(node):
                return node
            val = node.meta.get("val")
            if val is None:
                continue
            try:
                if free_unbacked_symbols(val):
                    return node
            except (AssertionError, TypeError):
                continue
        return None

    get_first_incompatible_cudagraph_node._trident_patched = True  # type: ignore[attr-defined]
    utils.get_first_incompatible_cudagraph_node = get_first_incompatible_cudagraph_node


def wrapper_decorator(mode: str, *, suite: Suite = "single") -> str:
    """Decorator line written onto the gems launch wrapper."""
    dynamic = suite == "multi"
    if mode == "triton":
        return ""
    if mode == "trident":
        # single: static, no outer shell (trident_dynamic=False in configure).
        # multi: dynamic=True; shell emitted when trident_dynamic=True.
        return f"@trident.jit(dynamic={dynamic})"
    if suite == "multi" and "cudagraph" in mode:
        raise ValueError(f"cudagraph not supported in multi suite: {mode}")
    options = []
    if "guard" in mode:
        options.append('"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe')
    if "cudagraph" in mode:
        options.append('"triton.cudagraphs": True')
    if "cpp_wrapper" in mode:
        options.append('"cpp_wrapper": True')
    args = ["fullgraph=True", f"dynamic={dynamic}"]
    if options:
        args.append("options={" + ", ".join(options) + "}")
    return f"@torch.compile({', '.join(args)})"


def apply_compile_runtime_flags(mode: str, *, n_shapes: int = 1) -> None:
    if "guard" in mode:
        torch._dynamo.config.install_free_tensors = True
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True
    if "cudagraph" in mode:
        patch_cudagraph_triton_meta()
    if "cpp_wrapper" in mode:
        torch._inductor.config.cpp_cache_precompile_headers = False
        if "cudagraph" in mode:
            allow_cpp_wrapper_cudagraph()
    if n_shapes > 1:
        torch._dynamo.config.recompile_limit = max(64, n_shapes + 8)
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = max(
                64, n_shapes + 8, int(torch._dynamo.config.cache_size_limit)
            )


def _preferred_dtype(dtypes):
    preferred = (torch.float16, torch.float32, torch.bfloat16, torch.int32, torch.bool)
    return next((dtype for dtype in preferred if dtype in dtypes), dtypes[0])


def _install_bench_config():
    from benchmark import base, conftest, consts

    config = conftest.BenchConfig()
    # Comprehensive mode preserves the selected benchmark input_fn semantics
    # (for example BLAS layout handling); shapes come only from OPS below.
    config.bench_level = consts.BenchLevel.COMPREHENSIVE
    base.Config = conftest.Config = config
    return config


def _resolve(spec: str):
    """Import ``package.module:attr`` (FlagGems repo root is on sys.path).

    Also supports ``harness:attr`` for local helpers defined in this file.
    """
    mod_name, attr = spec.rsplit(":", 1)
    if mod_name == "harness":
        return globals()[attr]
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


def _rms_norm_input_fn(shape, dtype, device):
    _, n = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    yield inp, (n,), weight


def _lt_scalar_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, 0.0


def _pow_scalar_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, 2.0


def _rsub_tensor_input_fn(shape, dtype, device):
    a = torch.randn(shape, dtype=dtype, device=device)
    b = torch.randn(shape, dtype=dtype, device=device)
    yield a, b


def _rsub_scalar_input_fn(shape, dtype, device):
    a = torch.randn(shape, dtype=dtype, device=device)
    yield a, 1.0


def _embedding_llm_input_fn(shape, dtype, device):
    # (num_tokens, num_embeddings, embedding_dim)
    n_tok, n_emb, dim = shape
    indices = torch.randint(0, n_emb, (n_tok,), device=device)
    weight = torch.randn((n_emb, dim), device=device, dtype=dtype)
    yield {"input": indices, "weight": weight},


def _pad_input_fn(shape, dtype, device):
    """Fixed pad params (gems bench uses random; we need cross-mode parity).

    Only pad the last two dims (H/W for NCHW). Padding every axis is not a
    realistic workload and produces oversized output tensors.
    """
    inp = torch.randn(shape, device=device, dtype=dtype)
    if inp.ndim < 2:
        pad_params = [1, 1]
    else:
        pad_params = [1, 1, 1, 1]  # left/right for W, then H
    yield inp, {"pad": pad_params, "mode": "constant", "value": 0.0}


def _conv3d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_d,
        input_h,
        input_w,
        out_c,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_d, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    yield {
        "input": inp,
        "weight": weight,
        "bias": None,
        "groups": groups,
        "stride": stride,
        "padding": padding,
    },


def _make_bench(op: str):
    from benchmark import base, consts

    meta = OPS[op]
    _install_bench_config()
    bench_kind = meta.get("bench", "unary_pointwise")
    float_dtypes = list(consts.FLOAT_DTYPES)

    if bench_kind == "unary_pointwise":
        bench = base.UnaryPointwiseBenchmark(
            op_name=meta["bench_op"],
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "binary_pointwise":
        bench = base.BinaryPointwiseBenchmark(
            op_name=meta["bench_op"],
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "unary_reduction":
        bench = base.UnaryReductionBenchmark(
            op_name=meta["bench_op"],
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "rms_norm":
        bench = base.GenericBenchmark2DOnly(
            op_name=meta["bench_op"],
            input_fn=_rms_norm_input_fn,
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "generic":
        bench = base.GenericBenchmark(
            op_name=meta["bench_op"],
            input_fn=_resolve(meta["input_fn"]),
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "generic_2d":
        bench = base.GenericBenchmark2DOnly(
            op_name=meta["bench_op"],
            input_fn=_resolve(meta["input_fn"]),
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "generic_excl_1d":
        bench = base.GenericBenchmarkExcluse1D(
            op_name=meta["bench_op"],
            input_fn=_resolve(meta["input_fn"]),
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "blas":
        bench = base.BlasBenchmark(
            op_name=meta["bench_op"],
            input_fn=_resolve(meta["input_fn"]),
            torch_op=meta["torch_op"],
            dtypes=float_dtypes,
        )
    elif bench_kind == "bench_cls":
        cls = _resolve(meta["bench_cls"])
        kwargs = dict(meta.get("bench_kwargs") or {})
        if "input_fn" in kwargs and isinstance(kwargs["input_fn"], str):
            kwargs["input_fn"] = _resolve(kwargs["input_fn"])
        if kwargs.get("dtypes") == "float16_32":
            kwargs["dtypes"] = [torch.float32, torch.float16]
        elif kwargs.get("dtypes") == "float16":
            kwargs["dtypes"] = [torch.float16]
        elif "dtypes" not in kwargs:
            kwargs["dtypes"] = float_dtypes
        if "torch_op" not in kwargs:
            kwargs["torch_op"] = meta["torch_op"]
        bench = cls(**kwargs)
    else:
        raise RuntimeError(f"unknown bench kind {bench_kind!r} for op={op}")

    # Deliberately do not call bench.set_shapes(): formal shapes live in OPS.
    bench.shapes = []
    return bench, meta


def load_selected_shapes(op: str, n: int | None) -> list:
    """Ordered explicit multi-shape family, capped at n."""
    shapes = list(OPS[op]["multi_shapes"])
    if n is None:
        return shapes
    return shapes[: max(0, min(n, len(shapes)))]


def load_inputs_for_shapes(op: str, shapes: list):
    """Build (shape, args, kwargs) bags; seed fixed so modes share inputs."""
    bench, meta = _make_bench(op)
    dtype = _preferred_dtype(bench.dtypes)
    bags = []
    for i, shape in enumerate(shapes):
        torch.manual_seed(0 + i)  # stable per shape index
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0 + i)
        bench.shapes = [shape]
        values = next(bench.get_input_iter(dtype))
        args, kwargs = bench.unpack_to_args_kwargs(values)
        bags.append((shape, args, kwargs))
    return bags, dtype, meta["torch_op"]


def load_single_inputs(op: str):
    """Build inputs for the explicit representative single shape."""
    shape = OPS[op]["single_shape"]
    bags, dtype, torch_op = load_inputs_for_shapes(op, [shape])
    shape, args, kwargs = bags[0]
    return shape, dtype, args, kwargs, torch_op


def resolve_call_fn(op: str):
    """Callable used inside timed loop (after patch_direct / configure_pointwise)."""
    meta = OPS[op]
    if meta.get("call") == "module_wrapper":
        module = importlib.import_module(f"flag_gems.ops.{meta['module']}")
        return getattr(module, meta["wrapper"])
    return meta["torch_op"]


def ensure_gems_include_lookup() -> None:
    """Repair FlagGems include lookup for ``@trident.jit`` wrappers.

    TridentGraphModule exposes ``__name__`` as a method, so
    ``FULL_CONFIG_BY_FUNC`` was keyed by the method object instead of the
    string name. ``use_gems(include=["linear"])`` then registered nothing
    and the harness silently timed native ATen. Mirror zsh's fix:
    assign a real string ``__name__`` and rebuild the lookup map.
    """
    import flag_gems

    rebuilt: dict[str, list] = {}
    for item in flag_gems._FULL_CONFIG:
        if not item or len(item) < 2:
            continue
        op_name, fn = item[0], item[1]
        name = getattr(fn, "__name__", None)
        if not isinstance(name, str):
            # Prefer the Aten / export name used in _FULL_CONFIG.
            try:
                fn.__name__ = str(op_name).split(".", 1)[0]
            except Exception:
                pass
            name = getattr(fn, "__name__", None)
            if not isinstance(name, str):
                name = str(op_name).split(".", 1)[0]
        rebuilt.setdefault(name, []).append(item)
        # Also index by the Aten key's base name when it differs.
        base = str(op_name).split(".", 1)[0]
        if base != name:
            rebuilt.setdefault(base, []).append(item)
    # Keep existing string aliases (e.g. softmax -> softmax_out).
    for key, value in list(flag_gems.FULL_CONFIG_BY_FUNC.items()):
        if isinstance(key, str) and key not in rebuilt:
            rebuilt[key] = list(value)
    flag_gems.FULL_CONFIG_BY_FUNC.clear()
    flag_gems.FULL_CONFIG_BY_FUNC.update(rebuilt)


def _strip_logger_debug(source: str) -> str:
    tree, lines = ast.parse(source), source.splitlines()
    for node in reversed(list(ast.walk(tree))):
        fn = (
            node.value.func
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            else None
        )
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "logger"
            and fn.attr == "debug"
        ):
            indent = lines[node.lineno - 1][
                : len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
            ]
            lines[node.lineno - 1 : node.end_lineno] = [indent + "pass"]
    return "\n".join(lines) + "\n"


def _ensure_import_torch(source: str) -> str:
    """ops that only import trident break when we rewrite to @torch.compile."""
    if re.search(r"(?m)^(import torch\b|from torch\b)", source):
        return source
    m = re.search(r"(?m)^(import |from )", source)
    if m:
        return source[: m.start()] + "import torch\n\n" + source[m.start() :]
    return "import torch\n\n" + source


def replace_direct_decorator(
    source: str, name: str, mode: str, *, suite: Suite = "single"
) -> str:
    pattern = rf"@trident\.jit(?:\([^)]*\))?\ndef {re.escape(name)}\("
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise RuntimeError(f"expected one @trident.jit wrapper for {name}, found {len(matches)}")
    decorator = wrapper_decorator(mode, suite=suite)
    replacement = f"{decorator}\ndef {name}(" if decorator else f"def {name}("
    source = source[: matches[0].start()] + replacement + source[matches[0].end() :]
    return source


def replace_direct_decorators(
    source: str, names: list[str] | tuple[str, ...], mode: str, *, suite: Suite = "single"
) -> str:
    for name in names:
        source = replace_direct_decorator(source, name, mode, suite=suite)
    if mode != "triton":
        source = _strip_logger_debug(source)
    # @torch.compile(...) needs torch in module scope at import/reload time.
    if mode.startswith("torch_compile"):
        source = _ensure_import_torch(source)
    return source


def replace_pad_codegen_decorator(
    source: str, mode: str, *, suite: Suite = "single"
) -> str:
    """Swap the decorator string emitted by pad's codegen template."""
    needle = 'code.writeline("@trident.jit(dynamic=False)")'
    if needle not in source:
        raise RuntimeError("pad codegen decorator writeline not found")
    decorator = wrapper_decorator(mode, suite=suite)
    if decorator:
        replacement = f"code.writeline({decorator!r})"
    else:
        # Triton: no host wrapper decorator on the generated pad function.
        replacement = "pass  # triton: no @trident.jit / @torch.compile on pad wrapper"
    return source.replace(needle, replacement, 1)


@contextlib.contextmanager
def patch_direct(
    op: str, mode: str, *, suite: Suite = "single", artifact_dir: Path | None = None
):
    """Rewrite one ops/<module>.py decorator for the duration of the with-block.

    Always restores the on-disk original in ``finally`` (including when
    ``write`` / ``reload`` fails). Also registers ``atexit`` so a soft
    process exit after a partial patch still tries to restore. SIGTERM is
    converted to SystemExit so Python unwinds this context before exiting.
    Hard ``SIGKILL`` can still leave a dirty file.
    """
    meta = OPS[op]
    if meta["kind"] != "direct":
        yield
        return
    path = SRC / "flag_gems/ops" / f"{meta['module']}.py"
    original = path.read_text()
    mod_name = f"flag_gems.ops.{meta['module']}"
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        # Prefer getting the file back even if reload blows up.
        try:
            path.write_text(original)
        finally:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    # File is restored; a stale in-process module is less bad
                    # than leaving a broken ops/*.py for the next worker.
                    pass

    atexit.register(restore)
    previous_sigterm = None

    def unwind_on_sigterm(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    try:
        try:
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, unwind_on_sigterm)
        except ValueError:
            # signal.signal is restricted to the main thread. Benchmark
            # workers run patch_direct on the main thread, but keep the
            # context manager usable in tests that do not.
            previous_sigterm = None
        if meta.get("patch") == "pad_codegen":
            patched = replace_pad_codegen_decorator(original, mode, suite=suite)
            if mode != "triton":
                patched = _strip_logger_debug(patched)
            if mode.startswith("torch_compile"):
                patched = _ensure_import_torch(patched)
        else:
            names = meta.get("wrappers")
            if names is None:
                names = (meta["wrapper"],)
            patched = replace_direct_decorators(original, names, mode, suite=suite)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "wrapper.py").write_text(patched)
        path.write_text(patched)
        # Inputs / bench may have already imported flag_gems; reload so the
        # patched decorator is what use_gems / resolve_call_fn actually call.
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        ensure_gems_include_lookup()
        yield
    finally:
        atexit.unregister(restore)
        restore()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        # Reload may have reintroduced method-valued __name__; repair again.
        try:
            ensure_gems_include_lookup()
        except Exception:
            pass


def skip_result_normalize_enabled(op: str, mode: str) -> bool:
    """Only pointwise + trident. Direct/other never."""
    return (
        POINTWISE_TRIDENT_SKIP_RESULT_NORMALIZE
        and OPS[op]["kind"] == "pointwise"
        and mode == "trident"
    )


def apply_trident_skip_env(env: dict, op: str, mode: str) -> None:
    """Explicit export / unset on a subprocess env (avoid parent-shell leakage).

    Pointwise+trident → set TRIDENT_SKIP_RESULT_NORMALIZE=1.
    Everything else → pop the key so a stale export cannot interfere.
    """
    if skip_result_normalize_enabled(op, mode):
        env["TRIDENT_SKIP_RESULT_NORMALIZE"] = "1"
    else:
        env.pop("TRIDENT_SKIP_RESULT_NORMALIZE", None)


def set_trident_skip_result_normalize(*, enabled: bool) -> None:
    """In-process mirror of apply_trident_skip_env."""
    if enabled:
        os.environ["TRIDENT_SKIP_RESULT_NORMALIZE"] = "1"
    else:
        os.environ.pop("TRIDENT_SKIP_RESULT_NORMALIZE", None)


def configure_pointwise(op: str, mode: str, *, suite: Suite = "single") -> None:
    meta = OPS[op]
    ensure_gems_include_lookup()
    if meta["kind"] != "pointwise":
        # Direct / other: never leave the skip-normalize export hanging.
        set_trident_skip_result_normalize(enabled=False)
        os.environ.pop("FLAGGEMS_POINTWISE_WRAPPER", None)
        return
    from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction

    set_trident_skip_result_normalize(
        enabled=skip_result_normalize_enabled(op, mode)
    )

    if mode == "triton":
        os.environ.pop("FLAGGEMS_POINTWISE_WRAPPER", None)
    else:
        os.environ["FLAGGEMS_POINTWISE_WRAPPER"] = wrapper_decorator(mode, suite=suite)

    for module_name, attr in meta["targets"]:
        module = importlib.import_module(module_name)
        target = getattr(module, attr)
        if not isinstance(target, PointwiseDynamicFunction):
            raise RuntimeError(f"{module_name}.{attr} is not PointwiseDynamicFunction")
        target.config.enable_trident_jit = mode != "triton"
        if suite == "single":
            # trident: static graph, no outer metadata shell.
            # torch.compile*: keep shell (trident_dynamic=True) for codegen path.
            target.config.trident_dynamic = mode != "trident"
        else:
            # multi: dynamic capture; shell needed for block-pointer stride_order.
            target.config.trident_dynamic = True
        target.overloads.clear()
        target._kernel_info_cache.clear()

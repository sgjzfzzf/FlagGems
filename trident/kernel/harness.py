#!/usr/bin/env python3
"""Shared FlagGems kernel-bench helpers (single-shape + multi-shape).

Formal protocol:
  - rounds = FORMAL_ROUNDS (5) for both suites
  - single: FORMAL_REPEATS (30) timed calls on the cheapest shape
    (no separate warmup; early samples = cold — slice later)
  - multi: each round times all selected shapes FORMAL_MULTI_PASSES
    times (≤ FORMAL_N_SHAPES); no FORMAL_REPEATS; no guard / cudagraph

Pointwise+trident only: TRIDENT_SKIP_RESULT_NORMALIZE=1 (timing hack).
Non-pointwise must explicitly clear that env so a parent export cannot leak.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import math
import os
import re
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
FORMAL_N_SHAPES = 32  # multi: ≤N shapes per pass
FORMAL_MULTI_PASSES = 3  # multi: full-shape passes per (round, mode)
# Pointwise+trident only (never for direct / other kinds).
POINTWISE_TRIDENT_SKIP_RESULT_NORMALIZE = True

# Catalog: Trident-wired FlagGems ops, driven by each op's existing gems Benchmark.
# Shapes: full gems bench pool → sort by cost ascending → take the cheapest
# min(FORMAL_N_SHAPES, len) (deterministic; identical across modes).
#
# pointwise enable_trident=True: abs, relu, sigmoid, add
# direct @trident.jit (or pad codegen): absolute, zeros_like, triu, tril, rms_norm_jit,
#   mm, bmm, addmm, addmv, linear, embedding, cumsum, sort, index_select, all,
#   cat, pad, conv2d, conv3d, conv_transpose1d, conv_transpose2d, flash_attention_forward
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
        "targets": (("flag_gems.ops.add", "add_func"),),
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
        "input_fn": "benchmark.test_cumsum:input_fn",
        "gems_include": ("cumsum",),
        "module": "cumsum",
        "wrapper": "cumsum",
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
        "bench": "bench_cls",
        "bench_cls": "benchmark.test_embedding:EmbeddingBenchmark",
        "bench_kwargs": {
            "op_name": "embedding",
            "input_fn": "benchmark.test_embedding:embedding_input_fn",
            "torch_op": torch.nn.functional.embedding,
            "dtypes": "float16_32",
        },
        "torch_op": torch.nn.functional.embedding,
        "gems_include": ("embedding",),
        "module": "embedding",
        "wrapper": "embedding",
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
    },
}

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


def shape_cost(shape: Any) -> float:
    """Cheapness key for selecting controllable N shapes (ascending).

    - Tensor-like shapes: product of positive ints
    - BLAS (B,M,N,K): M*N*K (ignore B for ranking when present as 4-tuple)
    - Skip bool (False subclasses int) and non-positive ints (padding/window 0).
    """
    if not isinstance(shape, (list, tuple)):
        return math.inf
    numbers = [x for x in shape if type(x) is int and x > 0]
    if not numbers:
        return math.inf
    if len(numbers) == 4:
        _b, m, n, k = numbers
        return float(m * n * k)
    return float(math.prod(numbers))


def select_shapes(shapes: list, n: int | None) -> list:
    """Take the n cheapest shapes (ascending cost). Same list for every mode."""
    ordered = sorted(shapes, key=shape_cost)
    if n is None:
        return ordered
    return ordered[: max(0, min(n, len(ordered)))]


def _preferred_dtype(dtypes):
    preferred = (torch.float16, torch.float32, torch.bfloat16, torch.int32, torch.bool)
    return next((dtype for dtype in preferred if dtype in dtypes), dtypes[0])


def _install_bench_config():
    from benchmark import base, conftest, consts

    config = conftest.BenchConfig()
    # Full gems pool (incl. set_more_shapes); we then keep only the
    # FORMAL_N_SHAPES cheapest by shape_cost.
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


def _pad_input_fn(shape, dtype, device):
    """Fixed pad params (gems bench uses random; we need cross-mode parity)."""
    inp = torch.randn(shape, device=device, dtype=dtype)
    rank = inp.ndim
    pad_params = [1, 1] * rank
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
    shape_file = str(ROOT / "benchmark/core_shapes.yaml")
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

    bench.set_shapes(shape_file)
    if "shapes" in meta:
        bench.shapes = [tuple(s) for s in meta["shapes"]]
    return bench, meta


def load_selected_shapes(op: str, n: int | None) -> list:
    """FlagGems bench shapes, cost-sorted, capped at n (shared across modes)."""
    bench, _ = _make_bench(op)
    return select_shapes(list(bench.shapes), n)


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


def load_smallest_inputs(op: str):
    """Single-shape: cheapest gems shape only."""
    shapes = load_selected_shapes(op, 1)
    bags, dtype, torch_op = load_inputs_for_shapes(op, shapes)
    shape, args, kwargs = bags[0]
    return shape, dtype, args, kwargs, torch_op


def resolve_call_fn(op: str):
    """Callable used inside timed loop (after patch_direct / configure_pointwise)."""
    meta = OPS[op]
    if meta.get("call") == "module_wrapper":
        module = importlib.import_module(f"flag_gems.ops.{meta['module']}")
        return getattr(module, meta["wrapper"])
    return meta["torch_op"]


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
    meta = OPS[op]
    if meta["kind"] != "direct":
        yield
        return
    path = SRC / "flag_gems/ops" / f"{meta['module']}.py"
    original = path.read_text()
    if meta.get("patch") == "pad_codegen":
        patched = replace_pad_codegen_decorator(original, mode, suite=suite)
        if mode != "triton":
            patched = _strip_logger_debug(patched)
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
    mod_name = f"flag_gems.ops.{meta['module']}"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    try:
        yield
    finally:
        path.write_text(original)
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


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

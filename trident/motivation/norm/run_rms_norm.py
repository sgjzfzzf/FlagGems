#!/usr/bin/env python3
"""Single-mode RMSNorm bare timing CLI (no Torch Profiler).

Invoked by bench_rms_norm.py once per (round, mode). Prints one JSON line to stdout.

Example:
  python run_rms_norm.py --mode trident --warmup 20 --repeats 10
  # JSON always can include host_us + e2e_us (default --metrics both)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rms_norm_trident import rms_norm_compile_entry, rms_norm_jit  # noqa: E402

MODES = (
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


def allow_cpp_wrapper_cudagraph() -> None:
    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    original = compile_fx.get_cpp_wrapper_config

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


def build(mode: str):
    if mode == "trident":
        return rms_norm_jit
    options = {}
    if "guard" in mode:
        torch._dynamo.config.install_free_tensors = True
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True
        options["guard_filter_fn"] = torch.compiler.keep_tensor_guards_unsafe
    if "cudagraph" in mode:
        options["triton.cudagraphs"] = True
        patch_cudagraph_triton_meta()
    if "cpp_wrapper" in mode:
        options["cpp_wrapper"] = True
        if "cudagraph" in mode:
            allow_cpp_wrapper_cudagraph()
    return torch.compile(
        rms_norm_compile_entry,
        fullgraph=True,
        dynamic=False,
        options=options or None,
    )


def timed(call):
    """Wall-clock around one op call.

    host_us: sync → call() returns (CPU dispatch / wrapper / launch; GPU usually still async).
    e2e_us:  same start → after cuda.synchronize() (host + GPU + wait).
    Recording host is one extra perf_counter read; it does not instrument the wrapper.
    """
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = call()
    host_us = (time.perf_counter_ns() - start) / 1e3
    torch.cuda.synchronize()
    e2e_us = (time.perf_counter_ns() - start) / 1e3
    return output, host_us, e2e_us


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--stage", choices=("cold", "warm"), default="warm")
    p.add_argument("--m", type=int, default=1024, help="Rows (e.g. B*S or B*S*H)")
    p.add_argument("--n", type=int, default=128, help="Norm dim (e.g. head_dim / hidden)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--metrics",
        choices=("e2e", "host", "both"),
        default="both",
        help="Which timings to keep in JSON (default: both host + e2e)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Per-job Triton/Inductor cache root (triton/ and inductor/ created under it)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_dir is not None:
        triton = args.cache_dir / "triton"
        inductor = args.cache_dir / "inductor"
        triton.mkdir(parents=True, exist_ok=True)
        inductor.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)

    torch.manual_seed(0)
    # x: (M, N), normalize over last dim — Qwen-like q/k norm uses N=head_dim.
    x = torch.randn(args.m, args.n, device="cuda", dtype=torch.float32)
    weight = torch.randn(args.n, device="cuda", dtype=torch.float32)
    eps = 1e-5
    normalized_shape = [args.n]
    ref = torch.nn.functional.rms_norm(x, (args.n,), weight=weight, eps=eps)
    fn = build(args.mode)

    def call():
        return fn(x, normalized_shape, weight, eps)

    warmup = args.warmup if args.stage == "warm" else 0
    for _ in range(warmup):
        call()
    if args.stage == "warm" and "guard" in args.mode:
        torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
    torch.cuda.synchronize()

    host_samples: list[float] = []
    e2e_samples: list[float] = []
    output = None
    for _ in range(args.repeats):
        output, host_us, e2e_us = timed(call)
        host_samples.append(host_us)
        e2e_samples.append(e2e_us)
    torch.testing.assert_close(output, ref, atol=1e-4, rtol=1e-3)

    # Always record both host and e2e; --metrics only selects what the bench aggregates.
    payload: dict = {
        "mode": args.mode,
        "stage": args.stage,
        "m": args.m,
        "n": args.n,
        "shape": [args.m, args.n],
        "warmup": warmup,
        "repeats": args.repeats,
        "metrics": args.metrics,
        "host_us": statistics.median(host_samples),
        "host_samples_us": host_samples,
        "e2e_us": statistics.median(e2e_samples),
        "e2e_samples_us": e2e_samples,
    }

    # One JSON object on stdout for the orchestrator to parse.
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()

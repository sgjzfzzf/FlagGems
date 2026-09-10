#!/usr/bin/env python3
"""Multi-shape FlagGems kernel timing CLI.

Invoked by bench_multi_shape.py once per (round, op, mode). Prints one JSON line.

Shapes: ordered application-shaped ``multi_shapes`` from harness.py, capped at
--n-shapes. Shapes in one op stay in the same dispatch/algorithm family.
Each round: --passes timed full passes over all selected shapes
(no separate warmup; early visits ≈ cold — slice later).

  host_us / e2e_us: median over all timed shape calls
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from harness import (  # noqa: E402
    FORMAL_MULTI_PASSES,
    FORMAL_N_SHAPES,
    MULTI_MODES,
    OPS,
    ROOT,
    apply_compile_runtime_flags,
    configure_pointwise,
    load_inputs_for_shapes,
    load_selected_shapes,
    patch_direct,
    resolve_call_fn,
    set_trident_skip_result_normalize,
    skip_result_normalize_enabled,
)

import torch  # noqa: E402


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = call()
    host_us = (time.perf_counter_ns() - start) / 1e3
    torch.cuda.synchronize()
    e2e_us = (time.perf_counter_ns() - start) / 1e3
    return output, host_us, e2e_us


def _shape_json(shape):
    if isinstance(shape, (list, tuple)):
        return list(shape)
    return shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=sorted(OPS), required=True)
    p.add_argument("--mode", choices=MULTI_MODES, required=True)
    p.add_argument("--n-shapes", type=int, default=FORMAL_N_SHAPES)
    p.add_argument(
        "--passes",
        type=int,
        default=FORMAL_MULTI_PASSES,
        help="Full timed passes over all selected shapes",
    )
    p.add_argument(
        "--metrics",
        choices=("e2e", "host", "both"),
        default="both",
    )
    p.add_argument("--cache-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "cudagraph" in args.mode or "guard" in args.mode:
        raise SystemExit(f"guard/cudagraph not supported in multi: {args.mode}")
    if args.passes < 1:
        raise SystemExit("--passes must be >= 1")

    cache = args.cache_dir.resolve()
    gems_cache = cache / "gems"
    triton_cache = cache / "triton"
    inductor_cache = cache / "inductor"
    for path in (gems_cache, triton_cache, inductor_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["FLAGGEMS_CACHE_DIR"] = str(gems_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    torch.manual_seed(0)
    shapes = load_selected_shapes(args.op, args.n_shapes)
    if not shapes:
        raise SystemExit(f"no shapes selected for op={args.op}")
    bags, dtype, _torch_op = load_inputs_for_shapes(args.op, shapes)
    apply_compile_runtime_flags(args.mode, n_shapes=len(shapes))

    kind = OPS[args.op]["kind"]
    skip_norm = skip_result_normalize_enabled(args.op, args.mode)
    set_trident_skip_result_normalize(enabled=skip_norm)

    with patch_direct(args.op, args.mode, suite="multi"):
        import flag_gems

        configure_pointwise(args.op, args.mode, suite="multi")
        call_fn = resolve_call_fn(args.op)

        gems_include = list(OPS[args.op]["gems_include"])
        with torch.no_grad(), flag_gems.use_gems(include=gems_include):
            torch.cuda.synchronize()
            host_samples: list[float] = []
            e2e_samples: list[float] = []
            shape_indices: list[int] = []
            pass_indices: list[int] = []
            for pass_i in range(args.passes):
                for idx, (shape, op_args, op_kwargs) in enumerate(bags):

                    def call(a=op_args, k=op_kwargs):
                        return call_fn(*a, **k)

                    _, host_us, e2e_us = timed(call)
                    host_samples.append(host_us)
                    e2e_samples.append(e2e_us)
                    shape_indices.append(idx)
                    pass_indices.append(pass_i)

    payload = {
        "suite": "multiple",
        "op": args.op,
        "kind": kind,
        "mode": args.mode,
        "n_shapes": len(shapes),
        "shapes": [_shape_json(s) for s in shapes],
        "dtype": str(dtype),
        "warmup": 0,
        "passes": args.passes,
        "metrics": args.metrics,
        "skip_result_normalize": skip_norm,
        "root": str(ROOT),
        "shape_indices": shape_indices,
        "pass_indices": pass_indices,
        "host_us": statistics.median(host_samples),
        "host_samples_us": host_samples,
        "e2e_us": statistics.median(e2e_samples),
        "e2e_samples_us": e2e_samples,
    }
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()

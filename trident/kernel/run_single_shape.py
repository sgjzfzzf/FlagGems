#!/usr/bin/env python3
"""Single-shape FlagGems kernel timing CLI.

Invoked by bench_single_shape.py once per (round, op, mode). Prints one JSON line.

Uses the explicit representative ``single_shape`` from harness.py.
No separate warmup: every call is timed (early samples ≈ cold; slice later).

  host_us: sync → call returns
  e2e_us:  same start → after cuda.synchronize()
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
    FORMAL_REPEATS,
    ROOT,
    SINGLE_MODES,
    OPS,
    apply_compile_runtime_flags,
    configure_pointwise,
    load_single_inputs,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=sorted(OPS), required=True)
    p.add_argument("--mode", choices=SINGLE_MODES, required=True)
    p.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    p.add_argument(
        "--metrics",
        choices=("e2e", "host", "both"),
        default="both",
    )
    p.add_argument("--cache-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    shape, dtype, op_args, op_kwargs, _torch_op = load_single_inputs(args.op)
    apply_compile_runtime_flags(args.mode, n_shapes=1)

    kind = OPS[args.op]["kind"]
    skip_norm = skip_result_normalize_enabled(args.op, args.mode)
    set_trident_skip_result_normalize(enabled=skip_norm)

    with patch_direct(args.op, args.mode, suite="single"):
        import flag_gems

        configure_pointwise(args.op, args.mode, suite="single")
        call_fn = resolve_call_fn(args.op)

        def call():
            return call_fn(*op_args, **op_kwargs)

        gems_include = list(OPS[args.op]["gems_include"])
        with torch.no_grad(), flag_gems.use_gems(include=gems_include):
            torch.cuda.synchronize()
            host_samples: list[float] = []
            e2e_samples: list[float] = []
            for _ in range(args.repeats):
                _, host_us, e2e_us = timed(call)
                host_samples.append(host_us)
                e2e_samples.append(e2e_us)

    payload = {
        "suite": "single",
        "op": args.op,
        "kind": kind,
        "mode": args.mode,
        "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
        "dtype": str(dtype),
        "warmup": 0,
        "repeats": args.repeats,
        "metrics": args.metrics,
        "skip_result_normalize": skip_norm,
        "root": str(ROOT),
        "host_us": statistics.median(host_samples),
        "host_samples_us": host_samples,
        "e2e_us": statistics.median(e2e_samples),
        "e2e_samples_us": e2e_samples,
    }
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()

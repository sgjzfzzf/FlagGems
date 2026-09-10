#!/usr/bin/env python3
"""Per-op single-shape probe (warm-tail median).

Example:
  python probe_single_shapes.py --op rms_norm --gpu 6
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from harness import (  # noqa: E402
    OPS,
    apply_compile_runtime_flags,
    configure_pointwise,
    ensure_gems_include_lookup,
    load_inputs_for_shapes,
    patch_direct,
    resolve_call_fn,
    set_trident_skip_result_normalize,
    skip_result_normalize_enabled,
)

import torch  # noqa: E402

DEFAULT_SHAPES = {
    "rms_norm": [
        (1, 64),
        (8, 64),
        (64, 64),
        (256, 64),
        (1, 512),
        (8, 512),
        (64, 512),
        (256, 512),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (128, 4096),
        (1, 8192),
        (8, 8192),
    ],
    "cumsum": [
        (1, 64),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (1, 4096),
        (8, 4096),
        (64, 4096),
        (1, 16384),
        (8, 16384),
        (1, 32768),
        (8, 32768),
        (64, 32768),
    ],
    "sort": [
        (1, 64),
        (64, 64),
        (1024, 64),
        (1, 512),
        (32, 512),
        (1, 1024),
        (32, 1024),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (1, 16384),
        (8, 16384),
    ],
    "mm": [
        (1, 1, 512, 512),
        (1, 8, 512, 512),
        (1, 32, 512, 512),
        (1, 1, 1024, 1024),
        (1, 8, 1024, 1024),
        (1, 1, 4096, 4096),
        (1, 8, 4096, 4096),
        (1, 32, 4096, 4096),
        (1, 1, 4096, 11008),
        (1, 8, 4096, 11008),
    ],
    "addmm": [
        (1, 1, 512, 512),
        (1, 8, 512, 512),
        (1, 32, 512, 512),
        (1, 1, 1024, 1024),
        (1, 8, 1024, 1024),
        (1, 1, 4096, 4096),
        (1, 8, 4096, 4096),
        (1, 32, 4096, 4096),
    ],
    "linear": [
        (1, 1, 3584, 18944),
        (1, 8, 3584, 18944),
        (1, 32, 3584, 18944),
        (1, 1, 3584, 3584),
        (1, 8, 3584, 3584),
        (1, 8, 18944, 3584),
        (1, 8, 4608, 3584),
        (1, 8, 152064, 3584),
    ],
    "cat": [
        (1, 64),
        (8, 64),
        (64, 64),
        (1, 512),
        (32, 512),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (128, 4096),
    ],
    "conv2d": [
        (1, 64, 8, 8, 64, 3, 3, 1, 1, 1),
        (1, 64, 14, 14, 64, 3, 3, 1, 1, 1),
        (4, 64, 14, 14, 64, 3, 3, 1, 1, 1),
        (1, 64, 28, 28, 64, 3, 3, 1, 1, 1),
        (4, 64, 28, 28, 64, 3, 3, 1, 1, 1),
        (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
        (4, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    ],
    "conv_transpose2d": [
        (1, 64, 8, 8, 64, 3, 3, 2, 1, 1),
        (4, 64, 8, 8, 64, 3, 3, 2, 1, 1),
        (1, 64, 16, 16, 64, 3, 3, 2, 1, 1),
        (4, 64, 16, 16, 64, 3, 3, 2, 1, 1),
        (1, 64, 32, 32, 64, 3, 3, 2, 1, 1),
        (4, 64, 32, 32, 64, 3, 3, 2, 1, 1),
        (1, 64, 16, 16, 64, 3, 3, 1, 1, 1),
        (4, 64, 16, 16, 64, 3, 3, 1, 1, 1),
    ],
    # zsh whitelist — Qwen25-ish token x hidden / FFN
    "silu": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
        (1, 18944),
        (8, 18944),
        (32, 18944),
        (128, 18944),
    ],
    "rsqrt": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
        (1, 8192),
        (8, 8192),
    ],
    "neg": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "add": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
        (8, 18944),
        (32, 18944),
    ],
    "lt": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "lt_scalar": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "pow": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "pow_scalar": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "rsub": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "rsub_scalar": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "floor_divide": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
    ],
    "masked_fill": [
        (1, 3584),
        (8, 3584),
        (32, 3584),
        (128, 3584),
        (8, 18944),
    ],
    "embedding": [
        (1, 152064, 3584),
        (8, 152064, 3584),
        (32, 152064, 3584),
        (128, 152064, 3584),
    ],
    "bmm": [
        (32, 1, 128, 128),
        (32, 8, 128, 128),
        (32, 32, 128, 128),
        (32, 128, 128, 128),
        (8, 8, 3584, 128),
        (8, 32, 3584, 128),
    ],
}


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    call()
    host_us = (time.perf_counter_ns() - start) / 1e3
    torch.cuda.synchronize()
    e2e_us = (time.perf_counter_ns() - start) / 1e3
    return host_us, e2e_us


def probe_one(
    op: str,
    shape,
    *,
    modes: tuple[str, ...],
    repeats: int,
    warm_tail: int,
) -> dict:
    bags, dtype, _ = load_inputs_for_shapes(op, [shape])
    _, args, kwargs = bags[0]
    out = {
        "op": op,
        "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
        "dtype": str(dtype),
        "modes": {},
    }
    for mode in modes:
        apply_compile_runtime_flags(mode, n_shapes=1)
        set_trident_skip_result_normalize(
            enabled=skip_result_normalize_enabled(op, mode)
        )
        cache = Path(tempfile.mkdtemp(prefix=f"probe-{op}-{mode}-"))
        for sub in ("gems", "triton", "inductor"):
            (cache / sub).mkdir()
        os.environ["FLAGGEMS_CACHE_DIR"] = str(cache / "gems")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        try:
            with patch_direct(op, mode, suite="single"):
                import flag_gems

                ensure_gems_include_lookup()
                configure_pointwise(op, mode, suite="single")
                fn = resolve_call_fn(op)

                def call(a=args, k=kwargs, f=fn):
                    return f(*a, **k)

                with torch.no_grad(), flag_gems.use_gems(
                    include=list(OPS[op]["gems_include"])
                ):
                    # Fail fast if include names did not register (silent ATen fallback).
                    registered = set(flag_gems.all_registered_ops())
                    registered_keys = set(flag_gems.all_registered_keys())
                    want = set(OPS[op]["gems_include"])
                    if not (
                        want <= registered
                        or any(
                            any(w == k or k.startswith(w + ".") for k in registered_keys)
                            for w in want
                        )
                    ):
                        raise RuntimeError(
                            f"use_gems include miss: want={sorted(want)} "
                            f"ops={sorted(registered)[:20]} keys={sorted(registered_keys)[:20]}"
                        )
                    torch.cuda.synchronize()
                    host_samples: list[float] = []
                    e2e_samples: list[float] = []
                    for _ in range(repeats):
                        host_us, e2e_us = timed(call)
                        host_samples.append(host_us)
                        e2e_samples.append(e2e_us)
            warm_h = host_samples[-warm_tail:]
            warm_e = e2e_samples[-warm_tail:]
            out["modes"][mode] = {
                "host_us": statistics.median(warm_h),
                "e2e_us": statistics.median(warm_e),
                "host_samples_us": host_samples,
                "e2e_samples_us": e2e_samples,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - probe should continue
            out["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def print_result(row: dict) -> None:
    shape = row["shape"]
    modes = row["modes"]
    base = modes.get("triton", {}).get("host_us")
    tr = modes.get("trident", {}).get("host_us")
    tc = modes.get("torch_compile", {}).get("host_us")
    cpp = modes.get("torch_compile_cpp_wrapper", {}).get("host_us")
    print(f"\n=== {row['op']} shape={shape} ===", flush=True)
    for mode, payload in modes.items():
        if payload.get("error"):
            print(f"  {mode:28} ERR {payload['error'][:160]}", flush=True)
            continue
        host = payload["host_us"]
        e2e = payload["e2e_us"]
        sp = (base / host) if base and host else float("nan")
        print(
            f"  {mode:28} host={host:8.1f}us ({sp:5.3f}x vs gems)  e2e={e2e:8.1f}us",
            flush=True,
        )
    if base and tr and tc:
        vs_gems = base / tr
        vs_tc = tc / tr
        vs_cpp = (cpp / tr) if cpp else float("nan")
        print(
            f"  => trident vs gems={vs_gems:.3f}x, vs torch_compile={vs_tc:.3f}x, "
            f"vs cpp={vs_cpp:.3f}x",
            flush=True,
        )
        if vs_gems > 1.05 and vs_tc > 1.05:
            print(
                "  *** CANDIDATE: trident beats gems and torch.compile ***",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", required=True, choices=sorted(OPS))
    p.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Python literal list of shapes, e.g. '[(64,64),(8,4096)]'",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--repeats", type=int, default=24)
    p.add_argument("--warm-tail", type=int, default=12)
    p.add_argument(
        "--modes",
        type=str,
        default="triton,torch_compile,torch_compile_cpp_wrapper,trident",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default under trident/kernel/probe_results)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    if args.shapes:
        shapes = list(eval(args.shapes, {"__builtins__": {}}))  # noqa: S307
    else:
        if args.op not in DEFAULT_SHAPES:
            raise SystemExit(
                f"no default shapes for {args.op}; pass --shapes explicitly"
            )
        shapes = DEFAULT_SHAPES[args.op]

    out_path = args.output or (
        _HERE / "probe_results" / f"probe_{args.op}_single.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for shape in shapes:
        row = probe_one(
            args.op,
            tuple(shape),
            modes=modes,
            repeats=args.repeats,
            warm_tail=args.warm_tail,
        )
        print_result(row)
        results.append(row)

    print("\n=== ranking by trident host vs gems ===", flush=True)
    ranked = []
    for row in results:
        gems = row["modes"].get("triton", {}).get("host_us")
        tr = row["modes"].get("trident", {}).get("host_us")
        tc = row["modes"].get("torch_compile", {}).get("host_us")
        if not (gems and tr):
            continue
        ranked.append(
            {
                "shape": row["shape"],
                "gems": gems,
                "trident": tr,
                "tc": tc,
                "vs_gems": gems / tr,
                "vs_tc": (tc / tr) if tc else None,
            }
        )
    ranked.sort(key=lambda x: x["vs_gems"], reverse=True)
    for item in ranked:
        mark = ""
        if item["vs_gems"] > 1.05 and item["vs_tc"] and item["vs_tc"] > 1.05:
            mark = " << beats gems+torch"
        print(
            f"  {item['shape']}: gems={item['gems']:.1f} trident={item['trident']:.1f} "
            f"tc={item['tc']} | vs_gems={item['vs_gems']:.3f} "
            f"vs_tc={item['vs_tc']}{mark}",
            flush=True,
        )

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

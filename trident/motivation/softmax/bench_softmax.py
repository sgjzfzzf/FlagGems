#!/usr/bin/env python3
"""Orchestrate Softmax bare benches: subprocess CLI per (round, mode), aggregate in arrays.

Aligned with motivation/norm/bench_rms_norm.py. For chrome traces use trace_softmax.py.

Example:
  python bench_softmax.py --rounds 5 --warmup 20 --repeats 10 --m 1024 --n 128
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUNNER = _HERE / "run_softmax.py"

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


def run_one(
    *,
    python: str,
    mode: str,
    stage: str,
    m: int,
    n: int,
    dim: int,
    warmup: int,
    repeats: int,
    metrics: str,
    cache_dir: Path,
    gpu: int,
) -> dict:
    cmd = [
        python,
        str(_RUNNER),
        "--mode",
        mode,
        "--stage",
        stage,
        "--m",
        str(m),
        "--n",
        str(n),
        "--dim",
        str(dim),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--metrics",
        metrics,
        "--cache-dir",
        str(cache_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")
    proc = subprocess.run(
        cmd,
        cwd=str(_HERE),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"run_softmax.py failed mode={mode} exit={proc.returncode}\n"
            f"cmd: {' '.join(cmd)}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"no stdout JSON from mode={mode}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"bad JSON from mode={mode}: {exc}") from exc


def summarize(by_mode: dict[str, dict[str, list[float]]], metrics: str) -> dict:
    keys = []
    if metrics in ("host", "both"):
        keys.append("host")
    if metrics in ("e2e", "both"):
        keys.append("e2e")

    header = f"{'mode':48}"
    for k in keys:
        header += f" {k + '_med':>10} {k + '_mean':>10} {k + '_std':>9}"
    print(header)

    payload = {}
    for mode in MODES:
        if mode not in by_mode:
            continue
        entry = {}
        line = f"{mode:48}"
        for k in keys:
            vals = by_mode[mode][k]
            med = statistics.median(vals)
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            entry[f"{k}_us_median"] = med
            entry[f"{k}_us_mean"] = mean
            entry[f"{k}_us_std"] = std
            entry[f"{k}_us_rounds"] = vals
            line += f" {med:10.3f} {mean:10.3f} {std:9.3f}"
        print(line)
        payload[mode] = entry
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--stage", choices=("cold", "warm"), default="warm")
    p.add_argument(
        "--metrics",
        choices=("e2e", "host", "both"),
        default="both",
    )
    p.add_argument("--modes", type=str, default=",".join(MODES))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "bench_results",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode: {m}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"host": [], "e2e": []}
    )

    print(
        f"[bench] rounds={args.rounds} warmup={args.warmup} repeats={args.repeats} "
        f"shape=({args.m},{args.n}) dim={args.dim} metrics={args.metrics} stage={args.stage}"
    )
    print(f"[bench] runner={_RUNNER}")

    for rd in range(1, args.rounds + 1):
        print(f"\n=== round {rd}/{args.rounds} ===", flush=True)
        for mode in modes:
            cache = args.output_dir / "cache" / f"round_{rd}" / f"{mode}_{args.stage}"
            print(f"[{mode}]", flush=True)
            result = run_one(
                python=args.python,
                mode=mode,
                stage=args.stage,
                m=args.m,
                n=args.n,
                dim=args.dim,
                warmup=args.warmup,
                repeats=args.repeats,
                metrics=args.metrics,
                cache_dir=cache,
                gpu=args.gpu,
            )
            parts = []
            if args.metrics in ("host", "both") and "host_us" in result:
                by_mode[mode]["host"].append(result["host_us"])
                parts.append(f"host_us={result['host_us']:.3f}")
            if args.metrics in ("e2e", "both") and "e2e_us" in result:
                by_mode[mode]["e2e"].append(result["e2e_us"])
                parts.append(f"e2e_us={result['e2e_us']:.3f}")
            print(f"  {' '.join(parts)} (median, n={args.repeats})", flush=True)

    print("\n=== summary (median / mean ± pstdev over rounds) ===")
    payload = {
        "config": {
            "rounds": args.rounds,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "m": args.m,
            "n": args.n,
            "dim": args.dim,
            "shape": [args.m, args.n],
            "stage": args.stage,
            "metrics": args.metrics,
            "modes": list(modes),
        },
        "modes": summarize(by_mode, args.metrics),
    }
    out = args.output_dir / "summary_mean.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

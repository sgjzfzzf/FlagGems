#!/usr/bin/env python3
"""Control script: per op run single-shape then multi-shape.

Env policy (explicit, avoids parent-shell leakage):
  - pointwise + trident → subprocess gets TRIDENT_SKIP_RESULT_NORMALIZE=1
  - every other (op, mode) → that key is unset in the child env

Defaults:
  - rounds=5 (both suites)
  - single: repeats=30 timed calls on the explicit representative shape
  - multi: each round = FORMAL_MULTI_PASSES full passes over exactly
    FORMAL_N_SHAPES explicit application-shaped dynamic-family shapes;
    no guard / cudagraph

Example:
  python bench_ops.py --ops abs,absolute --gpu 0
  python bench_ops.py --ops abs --rounds 1 --repeats 6 --n-shapes 3 \\
      --single-modes triton,trident --multi-modes triton,trident
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SINGLE = _HERE / "bench_single_shape.py"
_MULTI = _HERE / "bench_multi_shape.py"

sys.path.insert(0, str(_HERE))
from harness import (  # noqa: E402
    FORMAL_MULTI_PASSES,
    FORMAL_N_SHAPES,
    FORMAL_REPEATS,
    FORMAL_ROUNDS,
    MULTI_MODES,
    OPS,
    SINGLE_MODES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ops",
        type=str,
        default=",".join(OPS),
        help=f"Comma-separated ops (available: {', '.join(OPS)})",
    )
    p.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    p.add_argument(
        "--repeats",
        type=int,
        default=FORMAL_REPEATS,
        help="Single suite only: timed calls per (round, mode). Ignored by multi.",
    )
    p.add_argument(
        "--n-shapes",
        type=int,
        default=FORMAL_N_SHAPES,
        help="Multi suite: max shapes per pass.",
    )
    p.add_argument(
        "--multi-passes",
        type=int,
        default=FORMAL_MULTI_PASSES,
        help="Multi suite: full timed passes over all selected shapes per round.",
    )
    p.add_argument("--metrics", choices=("e2e", "host", "both"), default="both")
    p.add_argument(
        "--single-modes",
        type=str,
        default=",".join(SINGLE_MODES),
        help="Modes for single suite (dynamic=False, may include cudagraph/guard)",
    )
    p.add_argument(
        "--multi-modes",
        type=str,
        default=",".join(MULTI_MODES),
        help="Modes for multi suite (dynamic=True; no guard/cudagraph)",
    )
    p.add_argument(
        "--suites",
        type=str,
        default="single,multiple",
        help="Comma-separated: single and/or multiple",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "bench_results",
        help="Shared output dir (<op>-single.json / <op>-multiple.json)",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first op/suite failure (default: continue and summarize)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if <op>-single.json / <op>-multiple.json already exists",
    )
    return p.parse_args()


def run_suite(cmd: list[str]) -> None:
    print(f"[bench_ops] exec: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_HERE.parents[1]), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"suite failed exit={proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    ops = tuple(x.strip() for x in args.ops.split(",") if x.strip())
    suites = tuple(x.strip() for x in args.suites.split(",") if x.strip())
    for op in ops:
        if op not in OPS:
            raise SystemExit(f"unknown op: {op}")
    for suite in suites:
        if suite not in ("single", "multiple"):
            raise SystemExit(f"unknown suite: {suite} (use single,multiple)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[bench_ops] ops={ops} suites={suites} rounds={args.rounds} "
        f"single_repeats={args.repeats} multi_n_shapes<={args.n_shapes} "
        f"multi_passes={args.multi_passes} warmup=0 gpu={args.gpu} "
        f"output={args.output_dir} fail_fast={args.fail_fast} force={args.force}",
        flush=True,
    )
    print(
        "[bench_ops] env: TRIDENT_SKIP_RESULT_NORMALIZE only for "
        "pointwise+trident (explicit unset otherwise)",
        flush=True,
    )

    failures: list[str] = []
    skipped: list[str] = []
    for op in ops:
        kind = OPS[op]["kind"]
        print(f"\n######## op={op} kind={kind} ########", flush=True)

        if "single" in suites:
            out = args.output_dir / f"{op}-single.json"
            if out.is_file() and out.stat().st_size > 0 and not args.force:
                print(f"[bench_ops] skip {op}/single (exists: {out})", flush=True)
                skipped.append(f"{op}/single")
            else:
                cmd = [
                    args.python,
                    str(_SINGLE),
                    "--ops",
                    op,
                    "--modes",
                    args.single_modes,
                    "--rounds",
                    str(args.rounds),
                    "--repeats",
                    str(args.repeats),
                    "--metrics",
                    args.metrics,
                    "--output-dir",
                    str(args.output_dir),
                    "--gpu",
                    str(args.gpu),
                    "--python",
                    args.python,
                    "--force",
                ]
                try:
                    run_suite(cmd)
                except Exception as exc:
                    msg = f"{op}/single: {exc}"
                    print(f"[bench_ops] ERROR {msg}", flush=True)
                    failures.append(msg)
                    if args.fail_fast:
                        raise SystemExit(msg) from exc

        if "multiple" in suites:
            out = args.output_dir / f"{op}-multiple.json"
            if out.is_file() and out.stat().st_size > 0 and not args.force:
                print(f"[bench_ops] skip {op}/multiple (exists: {out})", flush=True)
                skipped.append(f"{op}/multiple")
            else:
                cmd = [
                    args.python,
                    str(_MULTI),
                    "--ops",
                    op,
                    "--modes",
                    args.multi_modes,
                    "--rounds",
                    str(args.rounds),
                    "--n-shapes",
                    str(args.n_shapes),
                    "--passes",
                    str(args.multi_passes),
                    "--metrics",
                    args.metrics,
                    "--output-dir",
                    str(args.output_dir),
                    "--gpu",
                    str(args.gpu),
                    "--python",
                    args.python,
                    "--force",
                ]
                try:
                    run_suite(cmd)
                except Exception as exc:
                    msg = f"{op}/multiple: {exc}"
                    print(f"[bench_ops] ERROR {msg}", flush=True)
                    failures.append(msg)
                    if args.fail_fast:
                        raise SystemExit(msg) from exc

    if skipped:
        print(f"\n[bench_ops] skipped {len(skipped)}:", flush=True)
        for s in skipped:
            print(f"  - {s}", flush=True)
    if failures:
        print(f"\n[bench_ops] done with {len(failures)} failure(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        raise SystemExit(1)
    print(f"\n[bench_ops] done → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Orchestrate single-shape FlagGems kernel benches.

Per (op, round, mode): fresh subprocess + temp compile cache.

No separate warmup — every call is timed (slice cold/warm later).

Writes:
  - bench-single.log
  - <op>-single.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUNNER = _HERE / "run_single_shape.py"

sys.path.insert(0, str(_HERE))
from harness import (  # noqa: E402
    FORMAL_REPEATS,
    FORMAL_ROUNDS,
    OPS,
    SINGLE_MODES,
    apply_trident_skip_env,
)


def run_one(
    *,
    python: str,
    op: str,
    mode: str,
    repeats: int,
    metrics: str,
    cache_dir: Path,
    gpu: int,
) -> dict:
    cmd = [
        python,
        str(_RUNNER),
        "--op",
        op,
        "--mode",
        mode,
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
    # Pointwise+trident: export skip-normalize. Others: explicit unset (no leak).
    apply_trident_skip_env(env, op, mode)
    if OPS[op]["kind"] != "pointwise":
        env.pop("FLAGGEMS_POINTWISE_WRAPPER", None)
    src = str((_HERE.parents[1] / "src").resolve())
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        cmd,
        cwd=str(_HERE.parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"run_single_shape.py failed op={op} mode={mode} exit={proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout[-4000:]}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"no stdout JSON from op={op} mode={mode}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"bad JSON from op={op} mode={mode}: {exc}") from exc


def format_log_line(result: dict, *, round_idx: int) -> str:
    parts = [
        f"round={round_idx}",
        f"suite=single",
        f"op={result.get('op')}",
        f"kind={result.get('kind')}",
        f"mode={result.get('mode')}",
        f"shape={result.get('shape')}",
        f"dtype={result.get('dtype')}",
        f"skip_result_normalize={result.get('skip_result_normalize')}",
    ]
    if "host_us" in result:
        parts.append(f"host_us={result['host_us']:.6f}")
    if "e2e_us" in result:
        parts.append(f"e2e_us={result['e2e_us']:.6f}")
    if "host_samples_us" in result:
        parts.append(f"host_samples_us={result['host_samples_us']}")
    if "e2e_samples_us" in result:
        parts.append(f"e2e_samples_us={result['e2e_samples_us']}")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    p.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    p.add_argument("--metrics", choices=("e2e", "host", "both"), default="both")
    p.add_argument(
        "--ops",
        type=str,
        default=",".join(OPS),
        help=f"Comma-separated ops (available: {', '.join(OPS)})",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=",".join(SINGLE_MODES),
        help="Comma-separated modes (single suite)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "bench_results",
        help="Writes bench-single.log and <op>-single.json",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if <op>-single.json already exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ops = tuple(x.strip() for x in args.ops.split(",") if x.strip())
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    for op in ops:
        if op not in OPS:
            raise SystemExit(f"unknown op: {op}")
    for mode in modes:
        if mode not in SINGLE_MODES:
            raise SystemExit(f"unknown single mode: {mode}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "bench-single.log"

    header = (
        f"[single] ops={ops} modes={len(modes)} rounds={args.rounds} "
        f"repeats={args.repeats} warmup=0 output={args.output_dir} force={args.force}"
    )
    print(header, flush=True)

    failures: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(header + "\n")
        log.flush()

        for op in ops:
            op_json = args.output_dir / f"{op}-single.json"
            if op_json.is_file() and op_json.stat().st_size > 0 and not args.force:
                msg = f"skip {op}/single (exists: {op_json})"
                print(f"\n=== {msg} ===", flush=True)
                log.write(f"=== {msg} ===\n")
                log.flush()
                continue

            print(f"\n=== op {op} (single) ===", flush=True)
            log.write(f"=== op {op} (single) ===\n")
            log.flush()

            op_runs: list[dict] = []
            try:
                for rd in range(1, args.rounds + 1):
                    print(f"-- round {rd}/{args.rounds} --", flush=True)
                    for mode in modes:
                        print(f"[{op}/{mode}]", flush=True)
                        try:
                            with tempfile.TemporaryDirectory(
                                prefix=f"fg-single-{op}-{mode}-"
                            ) as tmp:
                                result = run_one(
                                    python=args.python,
                                    op=op,
                                    mode=mode,
                                    repeats=args.repeats,
                                    metrics=args.metrics,
                                    cache_dir=Path(tmp).resolve(),
                                    gpu=args.gpu,
                                )
                            record = {
                                "round": rd,
                                "suite": "single",
                                "op": result.get("op"),
                                "kind": result.get("kind"),
                                "mode": result.get("mode"),
                                "shape": result.get("shape"),
                                "dtype": result.get("dtype"),
                                "warmup": 0,
                                "repeats": result.get("repeats"),
                                "metrics": result.get("metrics"),
                                "skip_result_normalize": result.get(
                                    "skip_result_normalize"
                                ),
                                "host_us": result.get("host_us"),
                                "host_samples_us": result.get("host_samples_us"),
                                "e2e_us": result.get("e2e_us"),
                                "e2e_samples_us": result.get("e2e_samples_us"),
                                "error": None,
                            }
                            op_runs.append(record)
                            line = format_log_line(result, round_idx=rd)
                            print(f"  {line}", flush=True)
                            log.write(line + "\n")
                            log.flush()
                        except Exception as exc:
                            msg = f"{op}/single round={rd} mode={mode}: {exc}"
                            print(f"  ERROR {msg}", flush=True)
                            log.write(f"ERROR {msg}\n")
                            log.flush()
                            failures.append(msg)
                            op_runs.append(
                                {
                                    "round": rd,
                                    "suite": "single",
                                    "op": op,
                                    "kind": OPS[op]["kind"],
                                    "mode": mode,
                                    "error": str(exc),
                                }
                            )
            except Exception as exc:
                msg = f"{op}/single: {exc}"
                print(f"ERROR {msg}", flush=True)
                log.write(f"ERROR {msg}\n")
                log.flush()
                failures.append(msg)

            payload = {
                "config": {
                    "suite": "single",
                    "rounds": args.rounds,
                    "warmup": 0,
                    "repeats": args.repeats,
                    "metrics": args.metrics,
                    "op": op,
                    "kind": OPS[op]["kind"],
                    "modes": list(modes),
                },
                "runs": op_runs,
                "failures": [f for f in failures if f.startswith(f"{op}/")],
            }
            op_json = args.output_dir / f"{op}-single.json"
            with op_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"wrote {op_json}", flush=True)
            log.write(f"wrote {op_json}\n")
            log.flush()

    print(f"\nwrote {log_path}", flush=True)
    if failures:
        print(f"[single] {len(failures)} failure(s) (continued)", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-shape *warm* e2e/host — same timing style as run_rms_norm.py / bench_rms_norm.py.

Per (mode, n_shapes, round): fresh process → compile/warmup all N shapes (untimed)
→ timed warm calls while rotating through those N shapes (cache hits + guard checks).

Compares e.g. N=1 (single shape, like before) vs N=16 vs N=32.
No cudagraph. Does NOT use skip_guard_eval_unsafe (need real cache/guard path).

Example:
  python bench_multi_shape_e2e.py --gpu 3 --n-shapes-list 1,16,32 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

_HERE = Path(__file__).resolve().parent

DEFAULT_MODES = (
    "torch_compile",
    "torch_compile_guard",
    "torch_compile_cpp_wrapper",
    "torch_compile_cpp_wrapper_guard",
    "trident",
)

PROBE = (1024, 128)


def make_shapes(n_shapes: int) -> list[tuple[int, int]]:
    """Distinct (M, N); N <= 4096. shapes[0] is always PROBE."""
    ms = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256,
          511, 512, 768, 1023, 1024, 1536, 2047, 2048, 3072, 4095, 4096]
    ns = [17, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
          448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
    out: list[tuple[int, int]] = []
    for i in range(max(len(ms), len(ns)) * 3):
        m = ms[i % len(ms)]
        n = ns[(i * 3 + 1) % len(ns)]
        if (m, n) not in out:
            out.append((m, n))
        if len(out) >= n_shapes:
            break
    if PROBE in out:
        out.remove(PROBE)
    out.insert(0, PROBE)
    return out[:n_shapes]


def timed(call):
    """Same as run_rms_norm.timed: host until return, e2e until cuda sync."""
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = call()
    host_us = (time.perf_counter_ns() - start) / 1e3
    torch.cuda.synchronize()
    e2e_us = (time.perf_counter_ns() - start) / 1e3
    return output, host_us, e2e_us


def worker_main(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(_HERE))
    from run_rms_norm import MODES, build

    if args.mode not in MODES:
        raise SystemExit(f"unknown mode: {args.mode}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.cache_dir is not None:
        triton = Path(args.cache_dir) / "triton"
        inductor = Path(args.cache_dir) / "inductor"
        triton.mkdir(parents=True, exist_ok=True)
        inductor.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)

    shapes = make_shapes(args.n_shapes)
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = max(64, len(shapes) + 8)
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = max(
            64, len(shapes) + 8, int(torch._dynamo.config.cache_size_limit)
        )

    fn = build(args.mode)
    bags = []
    for m, n in shapes:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        bags.append((x, [n], w, 1e-5))

    # Warmup / compile all shapes — NOT timed. Intentionally no skip_guard_eval_unsafe.
    for _ in range(args.warmup):
        for a in bags:
            fn(*a)
    torch.cuda.synchronize()

    # Timed warm path: rotate through the N cached shapes.
    host_samples: list[float] = []
    e2e_samples: list[float] = []
    for i in range(args.repeats):
        a = bags[i % len(bags)]
        _, host_us, e2e_us = timed(lambda a=a: fn(*a))
        host_samples.append(host_us)
        e2e_samples.append(e2e_us)

    payload = {
        "mode": args.mode,
        "stage": "warm",
        "n_shapes": len(shapes),
        "shapes": [list(s) for s in shapes],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "host_us": statistics.median(host_samples),
        "e2e_us": statistics.median(e2e_samples),
        "host_samples_us": host_samples,
        "e2e_samples_us": e2e_samples,
    }
    print(json.dumps(payload), flush=True)


def run_one(
    *,
    python: str,
    mode: str,
    n_shapes: int,
    warmup: int,
    repeats: int,
    gpu: int,
    cache_dir: Path,
) -> dict:
    cmd = [
        python,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--n-shapes",
        str(n_shapes),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--gpu",
        str(gpu),
        "--cache-dir",
        str(cache_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")
    proc = subprocess.run(
        cmd, cwd=str(_HERE), env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed mode={mode} n={n_shapes}\n"
            f"stderr:\n{proc.stderr[-4000:]}\nstdout:\n{proc.stdout[-2000:]}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"no stdout from worker mode={mode} n={n_shapes}")
    return json.loads(lines[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", type=str, default=None)
    p.add_argument("--n-shapes", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--n-shapes-list",
        type=str,
        default="1,16,32",
        help="Warm suites to compare (N cached shapes; 1 ≈ prior single-shape)",
    )
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20, help="Warmup passes over all N shapes")
    p.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Timed warm calls (rotate across the N shapes)",
    )
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "bench_multi_result",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not args.mode or not args.n_shapes:
            raise SystemExit("--worker needs --mode and --n-shapes")
        worker_main(args)
        return

    from run_rms_norm import MODES

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode: {m}")
        if "cudagraph" in m:
            raise SystemExit(f"cudagraph not supported: {m}")

    n_list = [int(x) for x in args.n_shapes_list.split(",") if x.strip()]
    max_n = max(n_list)
    all_shapes = make_shapes(max_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[multi-warm] gpu={args.gpu} n_shapes_list={n_list} rounds={args.rounds} "
        f"warmup={args.warmup} repeats={args.repeats}"
    )
    print(f"[multi-warm] modes={list(modes)}")
    print(f"[multi-warm] shapes[:max]={all_shapes}")

    results = []
    for mode in modes:
        for n in n_list:
            hosts, e2es = [], []
            for r in range(args.rounds):
                cache = args.output_dir / "cache" / f"{mode}_n{n}_r{r}"
                print(
                    f"  >> {mode}  n={n}  round={r + 1}/{args.rounds}",
                    flush=True,
                )
                one = run_one(
                    python=args.python,
                    mode=mode,
                    n_shapes=n,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    gpu=args.gpu,
                    cache_dir=cache,
                )
                hosts.append(one["host_us"])
                e2es.append(one["e2e_us"])
                print(
                    f"     host_us={one['host_us']:.3f}  e2e_us={one['e2e_us']:.3f}",
                    flush=True,
                )
            results.append(
                {
                    "mode": mode,
                    "n_shapes": n,
                    "shapes": [list(s) for s in all_shapes[:n]],
                    "host_us_rounds": hosts,
                    "e2e_us_rounds": e2es,
                    "host_us_median": statistics.median(hosts),
                    "e2e_us_median": statistics.median(e2es),
                    "host_us_mean": statistics.mean(hosts),
                    "e2e_us_mean": statistics.mean(e2es),
                    "host_us_std": statistics.pstdev(hosts) if len(hosts) > 1 else 0.0,
                    "e2e_us_std": statistics.pstdev(e2es) if len(e2es) > 1 else 0.0,
                }
            )

    print("\n=== warm median (µs) ===")
    print(
        "host = CPU until Python return; e2e = until cuda.synchronize(). "
        "N = #distinct shapes in Dynamo cache (timed calls rotate across them)."
    )
    # Human-readable header (two lines).
    line1 = f"{'mode':40}"
    line2 = f"{'':40}"
    for n in n_list:
        line1 += f"  {'N=' + str(n) + ' shapes':^17}"
        line2 += f"  {'host':>8} {'e2e':>8}"
    print(line1)
    print(line2)
    comparison = []
    for mode in modes:
        by_n = {r["n_shapes"]: r for r in results if r["mode"] == mode}
        line = f"{mode:40}"
        entry = {"mode": mode, "by_n": {}}
        for n in n_list:
            h = by_n[n]["host_us_median"]
            e = by_n[n]["e2e_us_median"]
            line += f"  {h:8.1f} {e:8.1f}"
            entry["by_n"][str(n)] = {"host_us_median": h, "e2e_us_median": e}
        if 1 in by_n and max_n in by_n:
            entry["host_ratio_nmax_over_1"] = (
                by_n[max_n]["host_us_median"] / by_n[1]["host_us_median"]
                if by_n[1]["host_us_median"]
                else None
            )
            entry["e2e_ratio_nmax_over_1"] = (
                by_n[max_n]["e2e_us_median"] / by_n[1]["e2e_us_median"]
                if by_n[1]["e2e_us_median"]
                else None
            )
        print(line)
        comparison.append(entry)

    payload = {
        "config": {
            "gpu": args.gpu,
            "n_shapes_list": n_list,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "modes": list(modes),
            "shapes_max": [list(s) for s in all_shapes],
            "skip_guard_eval_unsafe": False,
            "note": (
                "Same timed() as run_rms_norm (host + e2e). Warmup compiles all N "
                "shapes untimed; timed region rotates warm calls across those N "
                "cache entries. Compare n=1 vs larger N for multi-cache warm cost."
            ),
        },
        "results": results,
        "comparison": comparison,
    }
    out = args.output_dir / "summary_multi_shape_warm.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optional Chrome-trace capture for one Softmax mode (Torch Profiler).

Bare timing lives in run_softmax.py / bench_softmax.py — use this only for traces.

Example:
  python trace_softmax.py --mode trident --stage warm --output-dir ./profile_results
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_softmax import MODES, build  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--stage", choices=("cold", "warm"), default="warm")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--with-stack", action="store_true", default=True)
    p.add_argument("--no-stack", action="store_false", dest="with_stack")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.m, args.n, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.softmax(x, dim=args.dim)
    fn = build(args.mode)

    def call():
        return fn(x, args.dim)

    if args.stage == "warm":
        for _ in range(args.warmup):
            call()
        if "guard" in args.mode:
            torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
    torch.cuda.synchronize()

    config = torch._C._profiler._ExperimentalConfig(verbose=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=args.with_stack,
        experimental_config=config,
    ) as prof:
        with record_function(f"{args.stage}_e2e::{args.mode}"):
            output = call()
            torch.cuda.synchronize()
    torch.testing.assert_close(output, ref, atol=1e-4, rtol=1e-3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = args.output_dir / f"softmax_{args.mode}_{args.stage}_e2e_trace.json"
    prof.export_chrome_trace(str(trace))
    with trace.open("rb") as source, gzip.open(f"{trace}.gz", "wb", compresslevel=1) as target:
        shutil.copyfileobj(source, target)

    events = []
    for event in prof.events():
        if event.device_type != torch._C._autograd.DeviceType.CPU:
            continue
        events.append(
            {
                "name": event.name,
                "cpu_time_us": event.cpu_time_total,
                "self_cpu_time_us": event.self_cpu_time_total,
                "stack": event.stack,
            }
        )
    (args.output_dir / f"softmax_{args.mode}_{args.stage}_e2e_cpu_events.json").write_text(
        json.dumps(events, indent=2) + "\n"
    )
    for event in events:
        if event["name"] == f"{args.stage}_e2e::{args.mode}":
            print(f"{args.mode}: e2e_marker={event['cpu_time_us']/1e3:.3f} ms -> {trace}")
            break
    else:
        print(f"{trace}.gz")


if __name__ == "__main__":
    main()

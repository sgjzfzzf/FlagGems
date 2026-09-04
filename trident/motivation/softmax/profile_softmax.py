# Part of the Trident motivation experiments (FlagGems softmax).
# Pattern mirrors Trident/examples/profile_add.py.
#
# --measure profile: Torch Profiler traces (call structure)
# --measure bare:    uninstrumented timing; see --bare-metrics

import argparse
import gzip
import importlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

# Compiler comes from the launcher shell (source workspace/env/torch_env.sh).
# Do not override CC/CXX here.

import torch
from torch.profiler import ProfilerActivity, profile, record_function

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from softmax_trident import softmax_compile_entry, softmax_jit  # noqa: E402

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
    """Work around PyTorch unconditionally disabling this combination."""
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
    """Torch 2.12: cudagraph check crashes on Triton node meta dicts."""
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
        return softmax_jit
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
        softmax_compile_entry,
        fullgraph=True,
        dynamic=False,
        options=options or None,
    )


def timed(call):
    """Bare timing: host ends at Python return; e2e includes CUDA sync."""
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = call()
    host_us = (time.perf_counter_ns() - start) / 1e3
    torch.cuda.synchronize()
    e2e_us = (time.perf_counter_ns() - start) / 1e3
    return output, host_us, e2e_us


def run_profile(args, fn, call, ref) -> None:
    config = torch._C._profiler._ExperimentalConfig(verbose=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        experimental_config=config,
    ) as prof:
        with record_function(f"{args.stage}_e2e::{args.mode}"):
            output = call()
            torch.cuda.synchronize()
    torch.testing.assert_close(output, ref, atol=1e-4, rtol=1e-3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = args.output_dir / f"softmax_{args.mode}_{args.stage}_e2e_trace.json"
    prof.export_chrome_trace(str(trace))
    with trace.open("rb") as source, gzip.open(
        f"{trace}.gz", "wb", compresslevel=1
    ) as target:
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
            print(
                f"{args.mode}: e2e_marker={event['cpu_time_us']/1e3:.3f} ms -> {trace}"
            )
            break
    else:
        print(f"{trace}.gz")


def run_bare(args, call, ref) -> None:
    want_host = args.bare_metrics in ("host", "both")
    want_e2e = args.bare_metrics in ("e2e", "both")
    host_samples = []
    e2e_samples = []
    output = None
    for _ in range(args.repeats):
        output, host_us, e2e_us = timed(call)
        if want_host:
            host_samples.append(host_us)
        if want_e2e:
            e2e_samples.append(e2e_us)
    torch.testing.assert_close(output, ref, atol=1e-4, rtol=1e-3)

    payload = {
        "mode": args.mode,
        "stage": args.stage,
        "size": args.size,
        "warmup": args.warmup if args.stage == "warm" else 0,
        "repeats": args.repeats,
        "profiler": False,
        "bare_metrics": args.bare_metrics,
    }
    parts = []
    if want_host:
        host_us = statistics.median(host_samples)
        payload["host_us"] = host_us
        payload["host_us_mean"] = statistics.mean(host_samples)
        payload["host_samples_us"] = host_samples
        parts.append(f"host_us={host_us:.3f}")
    if want_e2e:
        e2e_us = statistics.median(e2e_samples)
        payload["e2e_us"] = e2e_us
        payload["e2e_us_mean"] = statistics.mean(e2e_samples)
        payload["e2e_samples_us"] = e2e_samples
        parts.append(f"e2e_us={e2e_us:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"softmax_{args.mode}_{args.stage}_bare.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{args.mode}: {' '.join(parts)} (median, n={args.repeats}) -> {out}")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    x = torch.randn(args.size, args.size, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.softmax(x, dim=1)
    fn = build(args.mode)

    def call():
        return fn(x, 1)

    if args.stage == "warm":
        for _ in range(args.warmup):
            call()
        if "guard" in args.mode:
            torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
    torch.cuda.synchronize()

    if args.measure == "bare":
        run_bare(args, call, ref)
    else:
        run_profile(args, fn, call, ref)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--stage", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--measure",
        choices=("profile", "bare"),
        default="profile",
        help="profile: Torch Profiler traces; bare: uninstrumented timing",
    )
    parser.add_argument(
        "--bare-metrics",
        choices=("e2e", "host", "both"),
        default="e2e",
        help="Which bare timings to record (default: e2e only)",
    )
    # One square case for now (matches gems core (64,64) when --size 64).
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iters before timing (all_modes.sh may override by measure)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Bare timed repeats (ignored for --measure profile)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

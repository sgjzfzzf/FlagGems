# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V2-Lite benchmark for the configured FlagGems whitelist."""

import argparse
import ast
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch

# Twelve pointwise entries plus bmm, linear, and embedding.
WHITELIST = (
    "lt_scalar",
    "lt",
    "rsqrt",
    "rsub_scalar",
    "rsub_tensor",
    "silu",
    "pow_tensor_tensor",
    "neg",
    "add",
    "masked_fill",
    "pow_tensor_scalar",
    "floor_divide",
    "bmm",
    "linear",
    "embedding",
)
MODES = ("torch", "gems", "torch_compile", "torch_compile_cpp_wrapper", "trident")
POINTWISE = {
    "add": ("add_func", "add_func_tensor_scalar", "add_func_scalar_tensor"),
    "div": (
        "floor_div_func",
        "floor_div_func_tensor_scalar",
        "floor_div_func_scalar_tensor",
    ),
    "lt": ("lt_func", "lt_func_scalar"),
    "masked_fill": ("masked_fill_kernel",),
    "neg": ("neg_func",),
    "pow": ("pow_func", "pow_func_tensor_scalar"),
    "rsqrt": ("rsqrt_func",),
    "rsub": ("rsub_func", "rsub_func_tensor_scalar"),
    "silu": ("silu_forward",),
}


def prompts(task, cache, limit):
    if task == "smoke":
        rows = [
            "Explain what a prime number is.",
            "Briefly explain photosynthesis.",
            "What causes ocean tides?",
            "Describe binary search.",
            "Why is the sky blue?",
            "What is a neural network?",
            "How does a compass work?",
            "Explain the water cycle.",
            "What is natural selection?",
            "How do batteries store energy?",
        ]
    else:
        from datasets import load_dataset

        name, subset = {
            "mmlu": ("cais/mmlu", "all"),
            "humaneval": ("openai/openai_humaneval", None),
            "gsm8k": ("openai/gsm8k", "main"),
        }[task]
        rows = load_dataset(name, subset, split="test", cache_dir=cache)
        if task == "mmlu":
            rows = [
                f"Question: {x['question']}\nChoices: {x['choices']}\nAnswer:"
                for x in rows
            ]
        elif task == "gsm8k":
            rows = [f"Solve step by step.\n{x['question']}\nAnswer:" for x in rows]
        else:
            rows = [x["prompt"] for x in rows]
    return rows[:limit] if limit else rows


def compile_source_copy():
    """Remove two known Inductor-incompatible Python details in a temp copy."""
    temp = tempfile.TemporaryDirectory(prefix="flaggems-compile-")
    src = Path(__file__).resolve().parents[3] / "src/flag_gems"
    dst = Path(temp.name) / "flag_gems"
    shutil.copytree(src, dst)
    for path in dst.rglob("*.py"):
        text = path.read_text()
        lines = text.splitlines()
        nodes = sorted(
            ast.walk(ast.parse(text)),
            key=lambda x: getattr(x, "lineno", 0),
            reverse=True,
        )
        for node in nodes:
            call = (
                node.value.func
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                else None
            )
            if (
                isinstance(call, ast.Attribute)
                and isinstance(call.value, ast.Name)
                and call.value.id == "logger"
                and call.attr in {"debug", "info", "warning"}
            ):
                indent = lines[node.lineno - 1][: -len(lines[node.lineno - 1].lstrip())]
                lines[node.lineno - 1 : node.end_lineno] = [indent + "pass"]
        text = "\n".join(lines).replace("ext.program_id(", "tl.program_id(")
        path.write_text(text.replace("ext.num_programs(", "tl.num_programs(") + "\n")
    sys.path.insert(0, temp.name)
    return temp


def configure(mode):
    """Replace the existing trident.jit decorator at the same wrapper boundary."""
    import trident

    if mode == "torch_compile_cpp_wrapper":
        # Clang 18 + GCC 13 headers can leave an unresolved std::string helper
        # in Inductor's global PCH; compiling the wrapper directly avoids it.
        torch._inductor.config.cpp_cache_precompile_headers = False

    trident_jit = trident.jit

    def jit(fn=None, **kwargs):
        if fn is None:
            return lambda f: jit(f, **kwargs)
        if mode == "gems":
            return fn
        if mode == "trident":
            compiled = trident_jit(fn, dynamic=True)
            compiled.__name__ = (
                fn.__name__
            )  # Preserve name-based Gems whitelist lookup.
            return compiled
        options = {"cpp_wrapper": True} if mode.endswith("cpp_wrapper") else {}
        return torch.compile(fn, fullgraph=True, dynamic=True, options=options)

    trident.jit = jit
    if mode.startswith("torch_compile"):
        torch._dynamo.config.recompile_limit = 4096
        torch._dynamo.config.accumulated_recompile_limit = 4096


def configure_pointwise(mode):
    """Enable the decorated wrapper only for compile and Trident modes."""
    enabled = mode != "gems"
    for module_name, names in POINTWISE.items():
        module = importlib.import_module(f"flag_gems.ops.{module_name}")
        for name in names:
            op = getattr(module, name)
            op.config.enable_trident_jit = enabled
            op.config.trident_dynamic = True
            op.overloads.clear()
            op._kernel_info_cache.clear()


@torch.inference_mode()
def decode(model, ids, count):
    start, end, token_ends = torch.cuda.Event(True), torch.cuda.Event(True), []
    current, past = ids, None
    start.record()
    for step in range(count):
        result = model(
            input_ids=current,
            attention_mask=torch.ones(
                (1, ids.shape[1] + step), dtype=torch.long, device=ids.device
            ),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = result.past_key_values
        current = result.logits[:, -1].argmax(-1, keepdim=True)
        token_ends.append(torch.cuda.Event(True))
        token_ends[-1].record()
    end.record()
    return start, end, token_ends


def request_times(events):
    start, end, tokens = events
    itl = [a.elapsed_time(b) for a, b in zip(tokens, tokens[1:])]
    return {
        "latency_ms": start.elapsed_time(end),
        "ttft_ms": start.elapsed_time(tokens[0]),
        "tpop_ms": sum(itl) / len(itl) if itl else 0.0,
        "itl_ms": itl,
    }


def stats(values):
    values = sorted(values)
    pick = lambda q: values[round((len(values) - 1) * q)] if values else None
    return {"p50_ms": pick(0.50), "p99_ms": pick(0.99)}


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument(
        "--task", choices=("smoke", "mmlu", "humaneval", "gsm8k"), default="humaneval"
    )
    p.add_argument("--samples", type=int, default=0, help="0 = full dataset")
    p.add_argument("--model-path", required=True)
    p.add_argument("--cache-dir")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, help="override the default 10%% split")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = prompts(args.task, args.cache_dir, args.samples)
    split = (
        args.warmup if args.warmup is not None else max(1, math.ceil(len(rows) * 0.1))
    )
    warm_rows, test_rows = rows[:split], rows[split:]
    if not test_rows:
        p.error("need at least two prompts")

    # Keep the temporary package alive until the benchmark process exits.
    args._source_copy = (
        compile_source_copy() if args.mode.startswith("torch_compile") else None
    )
    flag_gems = None
    if args.mode != "torch":
        configure(args.mode)
        import flag_gems

        configure_pointwise(args.mode)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    encode = lambda text: tokenizer(text, return_tensors="pt").input_ids.cuda()
    warm_inputs, test_inputs = map(encode, warm_rows), list(map(encode, test_rows))
    result = {
        "mode": args.mode,
        "whitelist": () if args.mode == "torch" else WHITELIST,
        "config": {
            "batch": 1,
            "warmup_ratio": None if args.warmup is not None else 0.1,
            "warmup": len(warm_rows),
            "measure": len(test_rows),
            "max_new_tokens": args.max_new_tokens,
            "chunk_size": args.chunk_size,
        },
        "chunks": [],
    }
    save(args.output, result)

    gems = (
        nullcontext() if args.mode == "torch" else flag_gems.use_gems(include=WHITELIST)
    )
    with gems, torch.inference_mode():
        if flag_gems is not None:
            registered = set(flag_gems.all_registered_ops())
            missing = set(WHITELIST) - registered
            if missing:
                raise RuntimeError(
                    f"Whitelist entries not registered: {sorted(missing)}"
                )
            result["registered_ops"] = sorted(registered)
            result["registered_keys"] = sorted(flag_gems.all_registered_keys())
            save(args.output, result)
            print(json.dumps({"registered_ops": result["registered_ops"]}), flush=True)
        for index, ids in enumerate(warm_inputs, 1):
            decode(model, ids, args.max_new_tokens)
            torch.cuda.synchronize()
            print(json.dumps({"warmup": index, "total": len(warm_rows)}), flush=True)

        for first in range(0, len(test_inputs), args.chunk_size):
            batch = test_inputs[first : first + args.chunk_size]
            print(
                json.dumps(
                    {"chunk_begin": first // args.chunk_size, "count": len(batch)}
                ),
                flush=True,
            )
            torch.cuda.synchronize()
            begin, finish = torch.cuda.Event(True), torch.cuda.Event(True)
            wall = time.perf_counter()
            begin.record()
            pending = [decode(model, ids, args.max_new_tokens) for ids in batch]
            finish.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - wall
            requests = [request_times(x) for x in pending]
            result["chunks"].append(
                {
                    "first": first,
                    "count": len(batch),
                    "wall_s": elapsed,
                    "gpu_ms": begin.elapsed_time(finish),
                    "input_tokens": sum(x.numel() for x in batch),
                    "requests": requests,
                }
            )

            chunks = result["chunks"]
            all_requests = [x for chunk in chunks for x in chunk["requests"]]
            itl = [x for request in all_requests for x in request["itl_ms"]]
            total_s = sum(x["wall_s"] for x in chunks)
            input_tokens = sum(x["input_tokens"] for x in chunks)
            output_tokens = len(all_requests) * args.max_new_tokens
            result["summary"] = {
                "latency": stats([x["latency_ms"] for x in all_requests]),
                "ttft": stats([x["ttft_ms"] for x in all_requests]),
                "tpop": stats([x["tpop_ms"] for x in all_requests]),
                "itl": stats(itl),
                "throughput_tokens_s": (input_tokens + output_tokens) / total_s,
                "output_tokens_s": output_tokens / total_s,
            }
            save(args.output, result)  # logging/flush is outside the timed interval
            print(
                json.dumps({"chunk": first // args.chunk_size, "wall_s": elapsed}),
                flush=True,
            )

    result["status"] = "complete"
    save(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

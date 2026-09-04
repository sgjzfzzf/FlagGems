#!/usr/bin/env python3
"""Compare Qwen generation with Trident dynamic JIT and origin FlagGems."""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# All FlagGems operator families that contain an @trident.jit implementation.
# Everything else stays on native ATen, including slicing and cache handling.
TRIDENT_JIT_OPS = [
    "absolute", "addmm", "addmv", "all", "bmm", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "cumsum", "embedding",
    "index_select", "linear", "mm",
    "pad", "tril", "triu", "zeros_like",
]

# Operators known to interfere with native Llama Python/Tensor indexing.
# The benchmark otherwise enables the complete FlagGems registry.
FLAGGEMS_BLACKLIST = [
    "slice.Tensor",
    "slice",
    "slice_backward",
    "slice_scatter",
]


def samples(task, limit):
    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = [f"Question: {r['question']}\nChoices: {r['choices']}\nAnswer:" for r in ds]
    elif task == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        rows = [r["prompt"] for r in ds]
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [f"Solve this problem step by step.\n{r['question']}\nAnswer:" for r in ds]
    else:
        raise ValueError(task)
    return rows[:limit] if limit else rows


@torch.inference_mode()
def fixed_decode(model, input_ids, max_new_tokens):
    """Greedy prefill + fixed one-token decode loop (no model.generate())."""
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_key_values = out.past_key_values
    next_token = out.logits.select(1, out.logits.shape[1] - 1).argmax(dim=-1, keepdim=True)
    result = input_ids
    for _ in range(max_new_tokens):
        result = torch.cat((result, next_token), dim=1)
        out = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        next_token = out.logits.select(1, out.logits.shape[1] - 1).argmax(dim=-1, keepdim=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/sunwenjia04/qwen3.5/Llama-3.2-3B")
    ap.add_argument("--flaggems-path", default="/home/wjsun/orginflaggems/FlagGems")
    ap.add_argument("--backend", choices=("trident", "flaggems"), required=True)
    ap.add_argument("--task", choices=("mmlu", "humaneval", "gsm8k", "all"), default="all")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    # Both backends are FlagGems operator sets.  The Trident JIT is used
    # internally by the operators in /home/wjsun/FlagGems; it must not wrap
    # the whole HuggingFace model.forward().
    if args.backend == "flaggems":
        sys.path.insert(0, os.path.join(args.flaggems_path, "src"))
    else:
        sys.path.insert(0, "/home/wjsun/newgems/FlagGems/src")
    import flag_gems
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="cuda", trust_remote_code=True
    ).eval()
    tasks = ("mmlu", "humaneval", "gsm8k") if args.task == "all" else (args.task,)
    result = {"backend": args.backend, "model": args.model_path, "tasks": {}}
    # Keep all unrelated ATen operators native.  In particular, Llama's
    # rotary embedding and cache bookkeeping use slicing/getitem paths that
    # should not be intercepted while comparing these kernels.
    ctx = flag_gems.use_gems(exclude=FLAGGEMS_BLACKLIST)
    with ctx, torch.inference_mode():
        for task in tasks:
            prompts = samples(task, args.limit)
            warm = min(len(prompts), math.ceil(len(prompts) * args.warmup_ratio))
            for p in prompts[:warm]:
                inp = tokenizer(p, return_tensors="pt").to("cuda")
                fixed_decode(model, inp["input_ids"], args.max_new_tokens)
            times, tokens = [], 0
            for p in prompts[warm:]:
                inp = tokenizer(p, return_tensors="pt").to("cuda")
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = fixed_decode(model, inp["input_ids"], args.max_new_tokens)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
                tokens += int(out.shape[-1] - inp["input_ids"].shape[-1])
            result["tasks"][task] = {"samples": len(prompts), "warmup": warm,
                "measured": len(times), "latency_mean_s": sum(times)/len(times) if times else None,
                "latency_p50_s": sorted(times)[len(times)//2] if times else None,
                "tokens_per_s": tokens/sum(times) if times and sum(times) else None}
    if args.output: args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


if __name__ == "__main__": main()

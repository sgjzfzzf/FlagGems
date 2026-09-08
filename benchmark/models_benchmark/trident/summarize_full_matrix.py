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

import argparse
import json
import re
import time
from pathlib import Path

TASKS = ("mmlu", "gsm8k", "humaneval")
MODES = ("torch", "gems", "torch_compile", "torch_compile_cpp_wrapper", "trident")
MODE_LABELS = {
    "torch": "Torch eager",
    "gems": "Gems",
    "torch_compile": "Gems + torch.compile",
    "torch_compile_cpp_wrapper": "Gems + torch.compile + C++ wrapper",
    "trident": "Gems + Trident",
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def count(log, pattern):
    try:
        return len(re.findall(pattern, log.read_text(errors="replace"), re.MULTILINE))
    except FileNotFoundError:
        return 0


def nested(data, *keys):
    for key in keys:
        data = data.get(key, {}) if isinstance(data, dict) else {}
    return data if isinstance(data, (int, float)) else None


def number(value):
    return "-" if value is None else f"{value:.3f}"


def collect(root):
    rows = []
    for task in TASKS:
        for mode in MODES:
            directory = root / task / mode
            result = read_json(directory / "results.json")
            exit_path = directory / "exit_code"
            exit_code = exit_path.read_text().strip() if exit_path.exists() else None
            if result.get("status") == "complete":
                status = "PASS"
            elif exit_code is not None:
                status = f"FAIL ({exit_code})"
            elif result:
                status = "RUNNING"
            else:
                status = "PENDING"
            rows.append(
                {
                    "task": task,
                    "mode": MODE_LABELS[mode],
                    "status": status,
                    "warm_done": count(directory / "run.log", r'^\{"warmup"'),
                    "warm_total": result.get("config", {}).get("warmup"),
                    "chunks_done": len(result.get("chunks", [])),
                    "measure": result.get("config", {}).get("measure"),
                    "summary": result.get("summary", {}),
                }
            )
    return rows


def write_status(root, rows):
    lines = [
        "# DeepSeek Full Matrix Status",
        "",
        "| Dataset | Mode | Status | Warmup | Measured chunks |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        warmup = f"{row['warm_done']}/{row['warm_total'] or '?'}"
        lines.append(
            f"| {row['task']} | {row['mode']} | {row['status']} | "
            f"{warmup} | {row['chunks_done']} |"
        )
    (root / "STATUS.md").write_text("\n".join(lines) + "\n")


def write_summary(root, rows):
    lines = [
        "# DeepSeek-V2-Lite Full Dataset Benchmark",
        "",
        "Configuration: batch size 1, chunk size 64, 128 generated tokens per prompt, "
        "and a separate 10% warmup split.",
        "",
        "| Dataset | Mode | Status | Warmup | Measure | Latency p50/p99 (ms) | "
        "TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Total tokens/s | Output tokens/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary = row["summary"]
        latency = "/".join(
            number(nested(summary, "latency", key)) for key in ("p50_ms", "p99_ms")
        )
        ttft = "/".join(
            number(nested(summary, "ttft", key)) for key in ("p50_ms", "p99_ms")
        )
        itl = "/".join(
            number(nested(summary, "itl", key)) for key in ("p50_ms", "p99_ms")
        )
        lines.append(
            f"| {row['task']} | {row['mode']} | {row['status']} | "
            f"{row['warm_total'] or '-'} | {row['measure'] or '-'} | {latency} | "
            f"{ttft} | {itl} | {number(summary.get('throughput_tokens_s'))} | "
            f"{number(summary.get('output_tokens_s'))} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize the full benchmark matrix.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    while True:
        rows = collect(args.root)
        write_status(args.root, rows)
        terminal = sum(row["status"].startswith(("PASS", "FAIL")) for row in rows)
        print(f"terminal={terminal}/15", flush=True)
        if terminal == 15:
            write_summary(args.root, rows)
            return
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

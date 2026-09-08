#!/usr/bin/env bash
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

set -uo pipefail

if (( $# < 2 )); then
    echo "Usage: RESULT_ROOT=... MODEL_PATH=... $0 GPU TASK:MODE [TASK:MODE ...]" >&2
    exit 2
fi

: "${RESULT_ROOT:?Set RESULT_ROOT to the output directory}"
: "${MODEL_PATH:?Set MODEL_PATH to the local DeepSeek-V2-Lite directory}"

gpu=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python}
dataset_cache=${DATASET_CACHE:-}
max_new_tokens=${MAX_NEW_TOKENS:-128}
chunk_size=${CHUNK_SIZE:-64}

export CUDA_VISIBLE_DEVICES="$gpu"

for job in "$@"; do
    task=${job%%:*}
    mode=${job#*:}
    job_dir="$RESULT_ROOT/$task/$mode"
    mkdir -p "$job_dir"

    if "$python" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' "$job_dir/results.json" 2>/dev/null; then
        echo "SKIP complete task=$task mode=$mode"
        continue
    fi

    export FLAGGEMS_CACHE_DIR="$job_dir/flaggems_cache"
    export TRITON_CACHE_DIR="$job_dir/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="$job_dir/inductor_cache"

    cache_args=()
    if [[ -n "$dataset_cache" ]]; then
        cache_args=(--cache-dir "$dataset_cache")
    fi

    echo "START gpu=$gpu task=$task mode=$mode time=$(date -u +%FT%TZ)"
    "$python" -u "$script_dir/benchmark_deepseek_whitelist.py" \
        --mode "$mode" \
        --task "$task" \
        --model-path "$MODEL_PATH" \
        "${cache_args[@]}" \
        --samples 0 \
        --max-new-tokens "$max_new_tokens" \
        --chunk-size "$chunk_size" \
        --output "$job_dir/results.json" \
        >"$job_dir/run.log" 2>&1
    code=$?
    echo "$code" >"$job_dir/exit_code"
    echo "END gpu=$gpu task=$task mode=$mode exit=$code time=$(date -u +%FT%TZ)"
done

echo "WORKER_DONE gpu=$gpu time=$(date -u +%FT%TZ)"

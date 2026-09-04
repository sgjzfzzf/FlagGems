#!/usr/bin/env bash
# Softmax motivation: all compile modes.
# Usage:
#   bash profile_softmax_all_modes.sh [--measure profile|bare] [--warmup N] [--repeats N] ...
# Env vars (MEASURE, WARMUP, REPEATS, ...) still work; CLI flags override them.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# motivation/<op> -> trident -> FlagGems -> workspace
workspace=$(cd "$root/../../../.." && pwd)
env_dir=${ENV_DIR:-$workspace/env}

python=${PYTHON:-python}
gpu=${GPU:-0}
measure=${MEASURE:-profile}
bare_metrics=${BARE_METRICS:-e2e}
size=${SIZE:-64}
stages_str=${STAGES:-warm}
output_override=${OUTPUT_DIR:-}
warmup_override=${WARMUP:-}
repeats_override=${REPEATS:-}

usage() {
    cat <<'EOF'
Usage: profile_softmax_all_modes.sh [options]

  --measure profile|bare     Measurement mode (default: profile; env MEASURE)
  --bare-metrics e2e|host|both  Bare metrics (default: e2e; env BARE_METRICS)
  --warmup N                 Warmup iters (bare default: 20; profile default: 3)
  --repeats N                Timed repeats (bare default: 10; profile default: 1)
  --size N                   Square size (default: 64; env SIZE)
  --stages "cold warm"       Stages (default: warm; env STAGES)
  --output-dir DIR           Results directory (env OUTPUT_DIR)
  --gpu N                    CUDA device id (default: 0; env GPU)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --measure)
            measure=$2
            shift 2
            ;;
        --bare-metrics)
            bare_metrics=$2
            shift 2
            ;;
        --warmup)
            warmup_override=$2
            shift 2
            ;;
        --repeats)
            repeats_override=$2
            shift 2
            ;;
        --size)
            size=$2
            shift 2
            ;;
        --stages)
            stages_str=$2
            shift 2
            ;;
        --output-dir)
            output_override=$2
            shift 2
            ;;
        --gpu)
            gpu=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

read -r -a stages <<<"$stages_str"

if [[ "$measure" == "bare" ]]; then
    output=${output_override:-$root/profile_results_bare}
    warmup=${warmup_override:-20}
    repeats=${repeats_override:-10}
else
    output=${output_override:-$root/profile_results}
    warmup=${warmup_override:-3}
    repeats=${repeats_override:-1}
fi

# Refresh profile/inductor toolchain every run (overrides any prior trident_env).
# shellcheck source=/dev/null
source "$env_dir/torch_env.sh"
echo "[env] CC=$CC CXX=$CXX (from $env_dir/torch_env.sh)"

modes=(
    torch_compile
    torch_compile_cudagraph
    torch_compile_guard
    torch_compile_guard_cudagraph
    torch_compile_cpp_wrapper
    torch_compile_cpp_wrapper_cudagraph
    torch_compile_cpp_wrapper_guard
    torch_compile_cpp_wrapper_guard_cudagraph
    trident
)

echo "[measure=$measure bare_metrics=$bare_metrics size=$size warmup=$warmup repeats=$repeats]"

# Fresh results each run (keep OUTPUT_DIR itself if it is a mount/symlink).
rm -rf "$output"
mkdir -p "$output"
cd "$root"
for mode in "${modes[@]}"; do
    for stage in "${stages[@]}"; do
        name=${mode}_${stage}
        cache=$output/cache/$name
        mkdir -p "$cache/triton" "$cache/inductor"
        echo "[$name]"
        # Re-source each job so a nested shell cannot keep a stale CC/CXX.
        # shellcheck source=/dev/null
        source "$env_dir/torch_env.sh"
        CUDA_VISIBLE_DEVICES=$gpu \
        TRITON_CACHE_DIR=$cache/triton \
        TORCHINDUCTOR_CACHE_DIR=$cache/inductor \
        "$python" "$root/profile_softmax.py" \
            --mode "$mode" --stage "$stage" --measure "$measure" \
            --bare-metrics "$bare_metrics" \
            --warmup "$warmup" --repeats "$repeats" --size "$size" \
            --output-dir "$output"
    done
done
echo "done. results in $output"

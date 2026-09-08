# DeepSeek Trident Benchmark

`benchmark_deepseek_whitelist.py` compares the following DeepSeek-V2-Lite
execution modes using a fixed FlagGems operator whitelist:

- Torch eager;
- FlagGems;
- FlagGems wrappers compiled with `torch.compile`;
- FlagGems wrappers compiled with `torch.compile(cpp_wrapper=True)`;
- FlagGems wrappers compiled with `trident.jit(dynamic=True)`.

Warmup and measurement use different prompts. After each chunk, the script
writes latency, throughput, TTFT, TPOP, and ITL metrics to JSON and flushes the
file outside the timed interval.

## Usage

Install the model dependencies and provide a local DeepSeek-V2-Lite path. For
example:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  benchmark/models_benchmark/trident/benchmark_deepseek_whitelist.py \
  --mode trident \
  --task humaneval \
  --model-path /path/to/DeepSeek-V2-Lite \
  --cache-dir /path/to/dataset-cache \
  --max-new-tokens 128 \
  --chunk-size 64 \
  --output results/trident.json
```

Available modes are `torch`, `gems`, `torch_compile`,
`torch_compile_cpp_wrapper`, and `trident`. Available datasets are `mmlu`,
`humaneval`, and `gsm8k`; `smoke` uses built-in prompts.

The `trident` mode means **FlagGems + Trident**: public Torch operators are
first dispatched to the selected FlagGems implementations, whose wrappers are
then captured with Trident.

## Full Matrix

`run_full_matrix_worker.sh` runs assigned `DATASET:MODE` jobs serially on one
GPU. Multiple workers can cover the full matrix in parallel. The following
example uses four GPUs and covers all three datasets and all five modes:

```bash
export RESULT_ROOT=/path/to/results
export MODEL_PATH=/path/to/DeepSeek-V2-Lite
export DATASET_CACHE=/path/to/dataset-cache
export PYTHON=python

benchmark/models_benchmark/trident/run_full_matrix_worker.sh 0 \
  mmlu:torch mmlu:trident &
benchmark/models_benchmark/trident/run_full_matrix_worker.sh 1 \
  mmlu:gems gsm8k:torch gsm8k:gems humaneval:torch humaneval:gems &
benchmark/models_benchmark/trident/run_full_matrix_worker.sh 2 \
  mmlu:torch_compile gsm8k:torch_compile gsm8k:trident \
  humaneval:torch_compile humaneval:trident &
benchmark/models_benchmark/trident/run_full_matrix_worker.sh 3 \
  mmlu:torch_compile_cpp_wrapper gsm8k:torch_compile_cpp_wrapper \
  humaneval:torch_compile_cpp_wrapper &
wait
```

Each job has an independent FlagGems, Triton, and Inductor cache. Completed
jobs are skipped when the worker is restarted. Generate `STATUS.md` and, once
all jobs are terminal, `SUMMARY.md` with:

```bash
python benchmark/models_benchmark/trident/summarize_full_matrix.py \
  --root "$RESULT_ROOT" --watch
```

## C++ Wrapper Issue and Fix

### Symptoms

Enabling the Inductor C++ wrapper exposed two consecutive errors.

First, Clang could not find the LLVM OpenMP runtime:

```text
/usr/bin/ld: cannot find -lomp: No such file or directory
clang++-18: error: linker command failed with exit code 1
```

After providing `libomp`, loading the generated C++ extension failed with an
undefined symbol:

```text
undefined symbol:
std::__cxx11::basic_string<...>::_M_init_local_buf()
```

### Root Causes

Inductor invokes Clang with `-fopenmp`, which requires an LLVM `libomp` version
matching the selected Clang toolchain. A PyTorch installation may contain GNU
`libgomp`, but it does not satisfy Clang's `-lomp` lookup and should not be
exposed through a fake `libomp` symlink.

The second failure comes from Inductor's global precompiled header (PCH) with
the tested Clang 18 and GCC 13 standard-library headers. The investigation
showed that:

- a minimal `std::string` extension compiled by Clang 18 loads successfully;
- using the Inductor global PCH reproduces the undefined symbol;
- the failure is therefore not caused by a FlagGems operator implementation.

### Fix

Install or unpack a `libomp` version matching the selected Clang toolchain, then
make its link and runtime paths visible before starting Python:

```bash
export LLVM_ROOT=/path/to/llvm-18
export LIBOMP_ROOT=/path/to/libomp-18
export CC="$LLVM_ROOT/bin/clang"
export CXX="$LLVM_ROOT/bin/clang++"
export LIBRARY_PATH="$LIBOMP_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$LIBOMP_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For `torch_compile_cpp_wrapper`, the benchmark also applies:

```python
torch._inductor.config.cpp_cache_precompile_headers = False
```

This disables only the incompatible global PCH. It does not disable the C++
wrapper or change operator execution. The main tradeoff is potentially longer
first-compilation time.

### Validation

The fix was validated with DeepSeek-V2-Lite, the benchmark's 15-entry FlagGems
whitelist, one independent warmup prompt, and one measured prompt:

- all 15 whitelist entries were registered;
- warmup completed;
- measured inference completed;
- the JSON result reported `status=complete`;
- the process exited with code zero.

This was a functionality smoke test and is not a performance result.

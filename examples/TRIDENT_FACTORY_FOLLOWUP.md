# Factory follow-up after renaming `device` to `dev`

Retested with Trident `2a308c14986fed1a0ac5a6b71949361e38fdc183`
(core/FFI built and installed), PyTorch `2.11.0+cu130`, and Python 3.12.
That Trident update changes the build configuration, not operator/guard logic.

The original name-collision reproduction in this PR is retained. The new script
uses thin `dev` adapters around the existing Gems implementations to get past
that collision. It removes existing Trident decoration before applying exactly
one boundary, and removes LibEntry from the inner kernels in the test process.
It does not copy the kernels, edit installed sources, or enable unrelated Gems
operators. These are targeted regression reproductions, not a claim that the
complete factory test suites pass.

## Reproduce

Run each command in a fresh process from the repository root:

```bash
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0  # choose an available GPU
export FLAGTREE_AABS=0

python examples/reproduce_trident_factory_followup.py arange --dtype float32
python examples/reproduce_trident_factory_followup.py arange --dtype float32 --torch-call
python examples/reproduce_trident_factory_followup.py ones --scalar
python examples/reproduce_trident_factory_followup.py zeros --scalar
python examples/reproduce_trident_factory_followup.py ones --dtype float32
python examples/reproduce_trident_factory_followup.py zeros --dtype float32
python examples/reproduce_trident_factory_followup.py eye
```

Use `--raw` for the same implementation without Trident, and `--static` to try
`dynamic=False`. The script checks both calls against a CPU reference and prints
the output type and specialization count. A zero exit status checks correctness;
it does **not** prove the second call reused a compiled specialization.

## Distinct issues, not one dtype error

| Case | Observation | Failure layer |
| --- | --- | --- |
| `arange`, direct Gems call, float32 | First result is `torch.Tensor`; second is `tvm_ffi.Tensor`, with specialization count staying at 1 | Cached output conversion |
| `arange`, Torch dispatch, same arguments | Both results are correct, but specialization count increases from 1 to 2 | Recompilation; the specific failing guard is not yet isolated |
| `ones` / `zeros`, shape `()` | `Failed to merge symbols`; `tvm_ffi.eq` receives native `i64` and `!tvm_ffi.int` | Empty-sequence guard IR verification |
| `ones` / `zeros`, shape `(8,)`, float32 | Cached result is a TVM FFI Tensor rather than a Torch Tensor | Cached output conversion |
| `eye(8, 8)` | `Unsupported input type <class 'torch.layout'>` | `torch.export` input handling |

### Empty-shape guard

The verifier diagnostic identifies this invalid comparison:

```text
%length = tvm_ffi.array.length(...) : (...) -> i64
%zero = tvm_ffi.constant.int(...) : () -> !tvm_ffi.int
%match = tvm_ffi.eq(%length, %zero) : (i64, !tvm_ffi.int) -> i1
```

In Trident's `python/trident/guards/ast.py`, `_build_not_sequence` produces a
native length, while the constant uses an FFI type; `_build_comparison(Eq)` does
not normalize the two before constructing `tvm_ffi.eq`. This is a guard **IR
type** mismatch, not an unsupported tensor dtype.

### Layout input

The `eye` adapter forwards the implementation's default `layout=torch.strided`.
Current Trident registers pytree nodes for `torch.device` and `torch.dtype` in
`python/trident/input.py`, but not for `torch.layout`. The latter is rejected by
the export input processing. Explicitly choosing float32 does not remove the
layout argument.

### First-call vs. cached return type

In `python/trident/backend.py`, `_build_sub_module` returns the FX Interpreter's
`warmup_result`; subsequent `__call__` invocations return the FFI executor result.
The latter is not uniformly converted back to Torch Tensor. TVM FFI's
`make_tensor_from_chandle` defaults to its own Tensor type when no framework
conversion context is available. Factory functions have no Tensor inputs; the
direct-call reproduction observes that fallback on the second invocation.

This can surface as an ATen `Unable to cast ... to Tensor` error, while this
script deliberately fails earlier with an explicit output-type assertion.

## Interpretation

Renaming `device` fixes the originally reported naming collision but does not
solve all subsequent failures. The earlier comment that the remaining issue was
`torch.dtype` describes the older version; the current results must distinguish
guard IR types, layout input support, output conversion, and cache reuse.
Repeated compilation can also hide the cached-output problem because every
compile returns a Torch `warmup_result`. No guard removal or extra copy is used
to mask these problems in this reproduction.

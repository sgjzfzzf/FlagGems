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

"""Reproduce factory failures after avoiding the device-name collision.

Run each case in a fresh process. This only changes objects in this process;
it does not modify installed sources or globally enable FlagGems operators.
"""

import argparse
import importlib
import json

import torch
import trident
from trident.backend import TridentGraphModule

from flag_gems.utils.libentry import LibEntry


def implementation(module_name, function_name):
    module = importlib.import_module(f"flag_gems.ops.{module_name}")
    # Remove LibEntry from the kernel, not the autotuner or Triton JIT.
    for key, value in list(vars(module).items()):
        if isinstance(value, LibEntry):
            setattr(module, key, value.fn)
    fn = getattr(module, function_name)
    return fn.fn if isinstance(fn, TridentGraphModule) else fn


arange_impl = implementation("arange", "arange_start")
ones_impl = implementation("ones", "ones")
zeros_impl = implementation("zeros", "zeros")
eye_impl = implementation("eye_m", "eye_m")


# Thin adapters give the captured graph an input named dev. The complete Gems
# implementation is inlined under one Trident boundary; no nested JIT is used.
def arange(start, end, step=1, *, dtype=None, layout=None, dev=None, pin_memory=None):
    return arange_impl(
        start, end, step, dtype=dtype, layout=layout, device=dev, pin_memory=pin_memory
    )


def ones(size, *, dtype=None, layout=None, dev=None, pin_memory=None):
    return ones_impl(
        size, dtype=dtype, layout=layout, device=dev, pin_memory=pin_memory
    )


def zeros(size, *, dtype=None, layout=None, dev=None, pin_memory=None):
    return zeros_impl(
        size, dtype=dtype, layout=layout, device=dev, pin_memory=pin_memory
    )


def eye(n, m, *, dtype=None, layout=torch.strided, dev=None, pin_memory=None):
    return eye_impl(n, m, dtype=dtype, layout=layout, device=dev, pin_memory=pin_memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=("arange", "ones", "zeros", "eye"))
    parser.add_argument(
        "--dtype", choices=("default", "float32", "int64"), default="default"
    )
    parser.add_argument(
        "--scalar", action="store_true", help="Use shape=() for ones/zeros"
    )
    parser.add_argument("--static", action="store_true", help="Use dynamic=False")
    parser.add_argument(
        "--raw", action="store_true", help="Run the same Gems code without Trident"
    )
    parser.add_argument(
        "--torch-call",
        action="store_true",
        help="Call through the matching ATen CUDA registration",
    )
    args = parser.parse_args()
    if args.scalar and args.op not in ("ones", "zeros"):
        parser.error("--scalar applies only to ones/zeros")
    dtype = None if args.dtype == "default" else getattr(torch, args.dtype)
    inputs = (
        (0, 64, 2)
        if args.op == "arange"
        else (8, 8) if args.op == "eye" else (() if args.scalar else (8,),)
    )
    raw = globals()[args.op]
    compiled = raw if args.raw else trident.jit(raw, dynamic=not args.static)
    call = compiled
    kwargs = dict(dtype=dtype, dev="cuda")
    if args.torch_call:

        def dispatch(*values, **options):
            options["dev"] = options.pop("device", None)
            return compiled(*values, **options)

        schema = {"arange": "arange.start_step", "eye": "eye.m"}.get(args.op, args.op)
        library = torch.library.Library("aten", "IMPL", "CUDA")
        library.impl(schema, dispatch)
        call = getattr(torch, args.op)
        kwargs = dict(dtype=dtype, device="cuda")
    if args.op == "eye":
        # Match the public wrapper forwarding its default layout explicitly.
        kwargs["layout"] = torch.strided
    print(json.dumps(vars(args)), flush=True)
    with torch.inference_mode():
        for iteration in range(2):
            output = call(*inputs, **kwargs)
            torch.cuda.synchronize()
            print(
                f"iteration={iteration} type={type(output)} "
                f"specializations={len(getattr(compiled, '_sub_modules', []))}",
                flush=True,
            )
            assert isinstance(
                output, torch.Tensor
            ), f"Expected torch.Tensor, got {type(output)}"
            expected = getattr(torch, args.op)(*inputs, dtype=dtype, device="cpu")
            torch.testing.assert_close(output.cpu(), expected)
    print(
        "Correct outputs; inspect specialization counts for recompilation.", flush=True
    )


if __name__ == "__main__":
    main()

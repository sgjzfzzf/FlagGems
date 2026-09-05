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

import logging

import torch
import trident
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)


@trident.jit
def _ones(size, *, dtype=None, layout=None, dev=None, pin_memory=None):
    logger.debug("GEMS ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dev is None:
        dev = torch.device(device_.name)

    out = torch.empty(size, device=dev, dtype=dtype)
    N = volume(size)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(dev):
        ones_kernel[grid](out, N, BLOCK_SIZE)
    return out


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _ones(
        size,
        dtype=dtype,
        layout=layout,
        dev=device,
        pin_memory=pin_memory,
    )

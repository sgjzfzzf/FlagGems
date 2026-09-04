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

from flag_gems import runtime

logger = logging.getLogger(__name__)


def conv_transpose1d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 1D transposed convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        output_padding: Output padding.
        dilation: Dilation.

    Returns:
        Output size of 1D transposed convolution.
    """
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@triton.autotune(
    configs=runtime.get_tuned_config("conv_transpose1d"),
    key=[
        "batch_size",
        "in_channels",
        "input_width",
        "out_channels",
        "out_width",
        "kernel_width",
        "stride_width",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose1d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    batch_size,
    input_width,
    out_channels,
    out_width,
    input_n_stride,
    input_c_stride,
    input_w_stride,
    weight_ic_stride,
    weight_oc_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_w_stride,
    in_channels: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_N_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Triton kernel for 1D transposed convolution forward pass.

    For transposed convolution:
    - input has shape (N, in_channels, in_width)
    - weight has shape (in_channels, out_channels/groups, kernel_width)
    - output has shape (N, out_channels, out_width)

    The output at position o is computed by summing contributions from all input
    positions i where the kernel at position k could have produced output at o:
    o = i * stride - padding + k * dilation
    => i = (o + padding - k * dilation) / stride (must be integer)
    """
    pid_n_ow = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Calculate batch and output width indices
    n_ow_offset = pid_n_ow * BLOCK_N_OW + tl.arange(0, BLOCK_N_OW)
    batch_idx = n_ow_offset // out_width
    out_w_idx = n_ow_offset % out_width

    # Output channel offset within this group
    out_channels_per_group = out_channels // groups
    # in_channels is already in_channels_per_group (passed from wrapper)
    in_channels_per_group = in_channels
    oc_offset = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    # Initialize accumulator
    accum = tl.zeros((BLOCK_N_OW, BLOCK_OC), dtype=tl.float32)

    # Pointers setup
    input_base = (
        input_pointer
        + (input_n_stride * batch_idx)[:, None]
        + (input_c_stride * pid_group * in_channels_per_group)
    )
    weight_base = (
        weight_pointer
        + (weight_ic_stride * pid_group * in_channels_per_group)
        + (weight_oc_stride * oc_offset)[None, :]
    )

    # Loop over input channels and kernel positions
    BLOCK_IC_COUNT = (in_channels_per_group + BLOCK_IC - 1) // BLOCK_IC
    for ic_k in range(BLOCK_IC_COUNT * kernel_width):
        ic_block = (ic_k // kernel_width) * BLOCK_IC
        k = ic_k % kernel_width

        ic_offset = ic_block + tl.arange(0, BLOCK_IC)

        # For transposed conv: out_w = in_w * stride - padding + k * dilation
        # So: in_w = (out_w + padding - k * dilation) / stride
        # We need in_w to be a valid integer index

        # Calculate the input position that contributes to this output
        numerator = out_w_idx + padding_width - k * dilation_width

        # Check if this is divisible by stride
        is_divisible = (numerator % stride_width) == 0
        in_w_idx = numerator // stride_width

        # Load input values
        curr_input_pointer = (
            input_base
            + (input_c_stride * ic_offset)[None, :]
            + (input_w_stride * in_w_idx)[:, None]
        )
        input_mask = (
            (batch_idx < batch_size)[:, None]
            & (ic_offset < in_channels_per_group)[None, :]
            & is_divisible[:, None]
            & (in_w_idx >= 0)[:, None]
            & (in_w_idx < input_width)[:, None]
        )
        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)

        # Load weight values
        # Weight shape: (in_channels, out_channels/groups, kernel_width)
        curr_weight_pointer = (
            weight_base
            + (weight_ic_stride * ic_offset)[:, None]
            + (weight_w_stride * k)
        )
        weight_mask = (ic_offset < in_channels_per_group)[:, None] & (
            oc_offset < out_channels_per_group
        )[None, :]
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

        # Accumulate: input_block is [BLOCK_N_OW, BLOCK_IC], weight_block is [BLOCK_IC, BLOCK_OC]
        accum += tl.dot(
            input_block.to(tl.float32), weight_block.to(tl.float32), allow_tf32=False
        )

    # Add bias if present
    bias_ptr = bias_pointer + pid_group * out_channels_per_group + oc_offset
    bias_mask = oc_offset < out_channels_per_group
    bias = tl.load(bias_ptr, mask=bias_mask, other=0.0).to(tl.float32)
    accum += bias[None, :]

    # Store output
    output_ptr = (
        output_pointer
        + (output_n_stride * batch_idx)[:, None]
        + (output_c_stride * (pid_group * out_channels_per_group + oc_offset))[None, :]
        + (output_w_stride * out_w_idx)[:, None]
    )
    output_mask = (
        (batch_idx < batch_size)[:, None]
        & (oc_offset < out_channels_per_group)[None, :]
        & (out_w_idx < out_width)[:, None]
    )
    tl.store(output_ptr, accum, mask=output_mask)


@trident.jit(dynamic=False)
def _conv_transpose1d_forward_impl(
    input_pointer,
    weight_pointer,
    bias_pointer,
    groups,
    stride,
    padding,
    output_padding,
    dilation,
):
    """Trident-JIT compiled conv_transpose1d forward: validation, parameter
    parsing, size derivation, contiguity, output allocation and kernel launch."""
    # Validation (constant-folded at trace time for valid shapes)
    assert input_pointer.ndim == 3, (
        f"Input must be 3D, received shape {input_pointer.shape}"
    )
    assert weight_pointer.ndim == 3, (
        f"Weights must be 3D, received shape {weight_pointer.shape}"
    )
    assert bias_pointer.ndim == 1, (
        f"Bias must be 1D, received shape {bias_pointer.shape}"
    )

    # Parse stride, padding, output_padding, dilation
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding

    if isinstance(output_padding, (list, tuple)):
        output_padding_width = output_padding[0]
    else:
        output_padding_width = output_padding

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    batch_size, in_channels, input_width = input_pointer.shape
    in_channels_weight, out_channels_per_group, kernel_width = weight_pointer.shape

    assert in_channels == in_channels_weight, (
        f"Input channels ({in_channels}) must match weight in_channels ({in_channels_weight})"
    )
    assert in_channels % groups == 0, (
        f"in_channels ({in_channels}) must be divisible by groups ({groups})"
    )

    out_channels = out_channels_per_group * groups

    assert bias_pointer.shape[0] == out_channels, (
        f"Bias shape ({bias_pointer.shape}) doesn't match out_channels ({out_channels})"
    )

    # Ensure contiguous tensors
    input_pointer = input_pointer.contiguous()
    weight_pointer = weight_pointer.contiguous()

    out_width = conv_transpose1d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )
    output_dtype = input_pointer.dtype
    output = torch.empty(
        (batch_size, out_channels, out_width),
        device=input_pointer.device,
        dtype=output_dtype,
    )

    in_channels_per_group = in_channels // groups
    grid = lambda META: (
        triton.cdiv(batch_size * out_width, META["BLOCK_N_OW"]),
        triton.cdiv(out_channels_per_group, META["BLOCK_OC"]),
        groups,
    )
    conv_transpose1d_forward_kernel[grid](
        input_pointer,
        weight_pointer,
        output,
        bias_pointer,
        batch_size,
        input_width,
        out_channels,
        out_width,
        *input_pointer.stride(),
        *weight_pointer.stride(),
        *output.stride(),
        in_channels_per_group,
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        groups=groups,
    )
    return output


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """
    Applies a 1D transposed convolution operator over an input signal.

    Args:
        input: Input tensor of shape (N, in_channels, L_in)
        weight: Filters of shape (in_channels, out_channels/groups, kernel_width)
        bias: Optional bias of shape (out_channels). Default: None
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides. Default: 0
        output_padding: Additional size added to output shape. Default: 0
        groups: Number of blocked connections. Default: 1
        dilation: Spacing between kernel elements. Default: 1

    Returns:
        Output tensor of shape (N, out_channels, L_out)
    """
    logger.debug("GEMS CONV_TRANSPOSE1D")

    # trident.jit cannot merge a None/optional tensor argument (a None-then-
    # tensor call order silently drops the tensor), so always pass a real bias
    # tensor and materialize zeros here when bias is absent.
    if bias is None:
        bias = torch.zeros(
            weight.shape[1] * groups, device=input.device, dtype=input.dtype
        )

    return _conv_transpose1d_forward_impl(
        input,
        weight,
        bias,
        groups,
        stride,
        padding,
        output_padding,
        dilation,
    )

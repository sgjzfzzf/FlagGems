"""Simplified RMSNorm Triton host wrapper (for paper schematic).

Dispatches to two kernels by a size threshold on N.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_oneshot_kernel(y_ptr, x_ptr, w_ptr, M, N, eps, BLOCK_SIZE: tl.constexpr):
    # one row per program; N fits in one block
    ...


@triton.jit
def rms_norm_loop_kernel(y_ptr, x_ptr, w_ptr, M, N, eps, TILE_N: tl.constexpr):
    # one row per program; tile over N when N is large
    ...


def rms_norm(x, weight, eps=1e-5):
    N = weight.numel()
    M = x.numel() // N
    y = torch.empty_like(x)

    if N <= 4096: # small-size problem
        BLOCK_SIZE = triton.next_power_of_2(N)
        rms_norm_oneshot_kernel[(M,)](
            y, x, weight, M, N, eps, BLOCK_SIZE=BLOCK_SIZE
        )
    else: # large-size problem
        TILE_N = 1024
        rms_norm_loop_kernel[(M,)](
            y, x, weight, M, N, eps, TILE_N=TILE_N
        )

    return y


# user call
x = torch.randn(1024, 128, device="cuda")
weight = torch.randn(128, device="cuda")
y = rms_norm(x, weight)

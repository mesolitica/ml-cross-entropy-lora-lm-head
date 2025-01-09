# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, overload

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping

def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    

def _cce_lse_forward_kernel(
    E,
    C,
    C_A,
    C_B,
    ALPHA,
    LSE,
    LA,
    Locks,
    Valids,
    softcap,
    B,
    V,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_cav,
    stride_cad,
    stride_lse_b,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_D:
            e = tl.load(e_ptrs)
            c = tl.load(c_ptrs)
        else:
            e = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0)
        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    logits = tl.where(v_mask[None, :], accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    o_mask = off_b < B
    if HAS_LA:
        logits = tl.where(o_mask[:, None], logits, 0.0)
        this_avg_logit = tl.sum(logits, 0) / B
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=v_mask)

    this_mx = tl.max(logits, axis=1)
    e = tl.exp(logits - this_mx[:, None])
    this_lse = this_mx + tl.log(tl.sum(e, axis=1))

    lse_ptrs = LSE + (stride_lse_b * off_b)

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.atomic_xchg(this_locks, 0)


_cce_lse_forward_kernel = triton.jit(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


@overload
def cce_lse_forward_kernel(
    e,
    c,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def cce_lse_forward_kernel(
    e,
    c,
    c_a,
    c_b,
    alpha,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def cce_lse_forward_kernel(
    e,
    c,
    c_a,
    c_b,
    alpha,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor: ...


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    c_a,
    c_b,
    alpha,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    V, D = c.shape
    lse = e.new_full((B,), -float("inf"), dtype=torch.float32)
    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None
    


    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        c_b,
        alpha,
        lse,  #
        logit_avg,
        locks,
        valids,
        softcap,
        B,
        V,
        D,  #
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        c_a.stride(0),
        c_b.stride(1),
        lse.stride(0),
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    if return_logit_avg:
        assert logit_avg is not None
        return lse, logit_avg
    else:
        return lse

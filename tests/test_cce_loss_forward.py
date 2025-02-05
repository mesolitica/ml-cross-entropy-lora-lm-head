# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")

import cut_cross_entropy
print(cut_cross_entropy)

def _loss(
    e: torch.Tensor,
    c: torch.Tensor,
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    alpha: float,
    targets: torch.Tensor,
    softcap: float | None,
    shift: bool,
) -> torch.Tensor:
    N, T = targets.size()
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    logits = e @ c.T
    left_e = e.to(c_a)
    right = alpha * ((e @ c_a.T) @ c_b.T)
    right = right.to(logits.dtype)
    logits = logits + right

    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = torch.nn.functional.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction="none"
    )

    return loss.view(N, T)


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce", "torch_compile"])
@pytest.mark.parametrize(
    "dtype,error_tol", [(torch.float32, 1e-5), (torch.float16, 1e-3), (torch.bfloat16, 1e-2)]
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("shape", [(256, 512, 128, 32), (252, 507, 128, 64), (252, 507, 123, 128)])
def test_loss_forward(
    impl: str,
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    shift: bool,
    invalids: bool,
    shape: tuple[int, int, int, int],
    alpha = 2.0,
):
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 256
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D, R = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)
    c_a = torch.randn((R, D), device="cuda", dtype=dtype)
    c_b = torch.randn((V, R), device="cuda", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    e = e.view(4, -1, D)

    targets = torch.randint(0, V, size=(N,), device="cuda")

    if invalids:
        inds = torch.randperm(len(targets), device="cuda")[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    targets = targets.view(e.size()[0:-1])

    gt = _loss(e.float(), c.float(), c_a.float(), c_b.float(), alpha, targets, softcap, shift)

    torch.set_float32_matmul_precision("highest" if dtype == torch.float32 else "high")
    ref = _loss(e, c, c_a, c_b, alpha, targets, softcap, shift)

    cce_loss = linear_cross_entropy(
        e, c, c_a, c_b, alpha, targets, softcap=softcap, shift=shift, reduction="none", impl=impl
    )

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_loss).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"

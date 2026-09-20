"""Fixed native cuBLASLt experiment for the two BF16 forward pipelines.

PyTorch 2.5.1 exposes preferred_blas_library("cublaslt"). Its BF16 GEMM
route uses BF16 A/B/C and CUBLAS_COMPUTE_32F, with no activation epilogue.
We change the vendor GEMM route, not any tensor dtype/layout or model cast.
No timing selector, audit-based fallback, shape table, or custom GEMM.
"""

import os

import torch
from torch.nn import functional as F


WORKSPACE_KIB = 32768


def configure_workspace():
    """Call before model loading/first Lt GEMM in the fresh engine process.

    The pinned C++ parser reads KiB and caches this value on its FIRST use.
    This permits a 32 MiB per-call workspace, not a promised peak-memory bound.
    No backend or precision flag is changed until an eligible linear runs.
    """
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = str(WORKSPACE_KIB)


def _eligible(x, weight):
    return (not torch.is_grad_enabled() and x.is_cuda and x.ndim == 2
            and weight.ndim == 2 and weight.device == x.device
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16)


def cublaslt_linear(x, weight):
    """Same bias-free linear and BF16 output; only change its BLAS preference.

    The process-wide preference is scoped to this synchronous host dispatch
    and restored even on failure. Kernels already queued/captured do not read
    it during execution. The engine serializes Python forward/capture calls;
    this wrapper is not intended for concurrent forwards in multiple threads.
    CPU, non-BF16, and gradient-enabled uses keep the original F.linear call.
    There are no casts, input/weight copies, or error-triggered vendor retries.
    """
    if not _eligible(x, weight):
        return F.linear(x, weight)
    previous = torch.backends.cuda.preferred_blas_library()
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        return F.linear(x, weight)
    finally:
        torch.backends.cuda.preferred_blas_library(previous)

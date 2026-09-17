# SPDX-License-Identifier: Apache-2.0
# Extracted from trainer/utils.py — CUDA utility functions needed by SP infra.
import torch

from utils.sp import envs
from utils.sp.logger import init_logger

logger = init_logger(__name__)


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `TRAINER_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.TRAINER_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable TRAINER_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return str(so_file)


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    if stream is not None:
        prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream | None:
    """
    replace `torch.cuda.current_stream()` with `current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    global _current_stream
    if _current_stream is None:
        _current_stream = torch.cuda.current_stream()
    return _current_stream

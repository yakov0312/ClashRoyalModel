"""Hardware setup shared by the trainers: NVIDIA (CUDA), AMD (ROCm/HIP),
Apple (MPS) or CPU.

PyTorch exposes ROCm GPUs through the same ``torch.cuda`` API, so "cuda"
means "a GPU" on both vendors; ``torch.version.hip`` tells them apart.

What runs where:
* The PPO update (big batched forward/backward) runs on the GPU; on NVIDIA
  with mixed precision (bf16 on Ampere+, else fp16 with loss scaling), on
  AMD in fp32 (see ``resolve_amp_dtype``).
* Rollouts pick one action at a time. At batch size 1 a GPU launch costs
  far more than the maths (measured ~17 ms per decision on a Radeon vs
  ~1.8 ms on one CPU core), so rollouts run on the CPU in parallel worker
  processes, one core each.
"""

from __future__ import annotations

import contextlib
import os

import torch

AMP_CHOICES = ("auto", "bf16", "fp16", "off")


def is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def resolve_torch_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested (--device cuda) but no CUDA/ROCm device is available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    # auto: NVIDIA/AMD GPU, then Apple GPU, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        backend = f"ROCm {torch.version.hip}" if is_rocm() else f"CUDA {torch.version.cuda}"
        return f"{torch.cuda.get_device_name(device)} ({backend})"
    return device.type


def configure_torch(device: torch.device) -> None:
    """Per-vendor speed settings; call once before building models."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    if is_rocm():
        # MIOpen's exhaustive kernel search re-runs for every new tensor shape
        # (PPO batches vary in length); the fast heuristic path avoids stalls.
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(device: torch.device, mode: str = "auto"):
    """torch dtype for autocast, or None to train in fp32.

    ``auto``: bf16 (else fp16) on NVIDIA; fp32 on AMD. On ROCm 7.14 with an
    RDNA4 card (RX 9060 XT) bf16 and fp16 training both hit a GPU memory
    fault inside hipBLASLt's fused-bias GEMM after a few iterations for some
    batch shapes (reproducible, independent of this code), so half precision
    on ROCm is opt-in only (``--amp bf16``/``fp16``).
    """
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "auto" and is_rocm():
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def autocast(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def make_grad_scaler(device: torch.device, amp_dtype) -> "torch.amp.GradScaler":
    """Loss scaling is only needed for fp16 (bf16 has fp32's range)."""
    return torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)


def default_num_workers() -> int:
    """Rollout worker processes: all cores but one (left for the learner)."""
    return max(1, (os.cpu_count() or 2) - 1)


def worker_process_init() -> None:
    """ProcessPoolExecutor initializer for rollout workers: die together with
    the trainer (Linux PR_SET_PDEATHSIG), so a crashed/killed trainer never
    leaves orphaned workers behind, and use one CPU thread each."""
    import signal
    import sys

    torch.set_num_threads(1)
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except OSError:
            pass

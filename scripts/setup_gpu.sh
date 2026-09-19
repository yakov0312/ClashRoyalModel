#!/usr/bin/env bash
# Set up the project with the right PyTorch build for this machine:
# NVIDIA (CUDA, e.g. vast.ai), AMD (ROCm) or CPU-only.
#
#   scripts/setup_gpu.sh            # auto-detect
#   scripts/setup_gpu.sh cuda|rocm|cpu
#
# Environment overrides:
#   USE_SYSTEM_PYTHON=1   install into the current python instead of .venv
#                         (useful on vast.ai images that already ship torch)
#   CUDA_INDEX / ROCM_INDEX / CPU_INDEX   PyTorch wheel index URLs
#
# Never run `uv sync` afterwards: torch is not in uv.lock (its build depends on
# the GPU), so sync would uninstall it. Use `uv pip install ...` instead.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-auto}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
CPU_INDEX="${CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ "$TARGET" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TARGET=cuda
  elif [[ -e /dev/kfd ]] || command -v rocminfo >/dev/null 2>&1; then
    TARGET=rocm
  else
    TARGET=cpu
  fi
fi
echo "==> target: $TARGET"

# --- python environment ---------------------------------------------------
if [[ "${USE_SYSTEM_PYTHON:-0}" == "1" ]]; then
  PY="$(command -v python3)"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  uv venv .venv
  PY=".venv/bin/python"
else
  python3 -m venv .venv
  PY=".venv/bin/python"
fi

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install --python "$PY")
else
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  PIP=("$PY" -m pip install)
fi
echo "==> python: $PY"

# --- PyTorch ----------------------------------------------------------------
current="$("$PY" - <<'EOF' 2>/dev/null || true
import torch
backend = "rocm" if torch.version.hip else ("cuda" if torch.version.cuda else "cpu")
ok = torch.cuda.is_available() if backend != "cpu" else True
print(backend if ok else "")
EOF
)"
if [[ "$current" == "$TARGET" ]]; then
  echo "==> keeping installed torch ($current build already works)"
else
  case "$TARGET" in
    cuda) INDEX="$CUDA_INDEX" ;;
    rocm) INDEX="$ROCM_INDEX" ;;
    *)    INDEX="$CPU_INDEX" ;;
  esac
  echo "==> installing torch from $INDEX"
  "${PIP[@]}" --index-url "$INDEX" torch
fi

# --- project dependencies (from pyproject.toml; the code runs from the repo root)
mapfile -t DEPS < <("$PY" -c '
import re
text = open("pyproject.toml").read()
block = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S).group(1)
print("\n".join(re.findall(r"\"([^\"]+)\"", block)))
')
"${PIP[@]}" "${DEPS[@]}"

# --- report -------------------------------------------------------------------
"$PY" - <<'EOF'
import os
from clasher.rl.accel import default_num_workers, describe_device, resolve_amp_dtype, resolve_torch_device
device = resolve_torch_device("auto")
print(f"==> device: {describe_device(device)}")
print(f"==> mixed precision: {resolve_amp_dtype(device)}")
print(f"==> rollout workers (auto): {default_num_workers()} of {os.cpu_count()} cores")
EOF

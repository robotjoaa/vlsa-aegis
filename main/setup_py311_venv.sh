#!/usr/bin/env bash
# =============================================================================
# Recreate main/.venv with Python 3.11 and all required dependencies.
#
# Priority order:
#   1. OSCBF (jax==0.4.30 + cbfpy + qpax + pybullet + trimesh)
#   2. oscbf package (editable install)
#   3. safelibero / robosuite
#   4. psfjax (no install needed — imported directly from repo root)
#   5. PyTorch 2.x + vision tools
#   6. Evaluation utilities
#
# Usage:
#   conda activate libero      # optional — provides conda context
#   bash main/setup_py311_venv.sh
#
# After completion:
#   source main/.venv/bin/activate
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/main/.venv"

echo "=== Repo root: $REPO_ROOT ==="
echo "=== Target venv: $VENV ==="

# ---------------------------------------------------------------------------
# 1. Recreate venv with Python 3.11 (uv)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [1/8] Creating Python 3.11 venv..."
uv venv "$VENV" --python 3.11 --seed --clear
source "$VENV/bin/activate"
echo "    Python: $(python --version)"

# ---------------------------------------------------------------------------
# 2. OSCBF — JAX 0.4.30 with CUDA 12 (PRIORITY)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [2/8] Installing JAX 0.4.30 (CUDA 12)..."
pip install "jax[cuda12_pip]==0.4.30" "jaxlib==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ">>> [2/8] Installing other OSCBF deps..."
pip install \
    "cbfpy==0.0.1" \
    "qpax" \
    "pybullet>=3.2.7" \
    "trimesh" \
    "networkx" \
    "lxml"

echo ">>> [2/8] Installing oscbf package (editable)..."
pip install -e "$REPO_ROOT/oscbf"

# ---------------------------------------------------------------------------
# 3. Simulation / Robot (safelibero + robosuite)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [3/8] Installing simulation dependencies..."
pip install \
    "mujoco==3.2.3" \
    "robosuite==1.4.1" \
    "bddl==1.0.1" \
    "h5py" \
    "gymnasium" \
    "pyquaternion" \
    "easydict" \
    "pyopengl" \
    "glfw"

echo ">>> [3/8] Installing safelibero (libero) package..."
pip install -e "$REPO_ROOT/safelibero"

# ---------------------------------------------------------------------------
# 4. Core scientific + QP solvers
# ---------------------------------------------------------------------------
echo ""
echo ">>> [4/8] Installing scientific stack..."
pip install \
    "numpy>=1.24,<2.0" \
    "scipy>=1.10" \
    "matplotlib" \
    "opencv-python" \
    "open3d==0.19.0" \
    "osqp" \
    "cvxpy"

# ---------------------------------------------------------------------------
# 5. PyTorch 2.x + CUDA 12.4
# ---------------------------------------------------------------------------
echo ""
echo ">>> [5/8] Installing PyTorch 2.x (CUDA 12.4)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install \
    "timm" \
    "einops" \
    "transformers>=4.21" \
    "tokenizers"

# ---------------------------------------------------------------------------
# 6. Vision / evaluation tools
# ---------------------------------------------------------------------------
echo ""
echo ">>> [6/8] Installing vision / evaluation tools..."
pip install \
    "groundingdino-py==0.4.0" \
    "supervision==0.6.0" \
    "pycocotools"

# ---------------------------------------------------------------------------
# 7. Utilities
# ---------------------------------------------------------------------------
echo ""
echo ">>> [7/8] Installing utilities..."
pip install \
    "tqdm" \
    "tyro" \
    "wandb" \
    "imageio" \
    "imageio-ffmpeg" \
    "pillow" \
    "pandas" \
    "scikit-learn" \
    "absl-py" \
    "huggingface-hub" \
    "safetensors" \
    "hydra-core" \
    "omegaconf" \
    "rich" \
    "retrying" \
    "websockets" \
    "requests"

# ---------------------------------------------------------------------------
# 8. OpenPI client (editable)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [8/8] Installing openpi-client (editable)..."
OPENPI_CLIENT="$REPO_ROOT/openpi/packages/openpi-client"
if [ -d "$OPENPI_CLIENT" ]; then
    pip install -e "$OPENPI_CLIENT"
else
    echo "    WARNING: openpi-client not found at $OPENPI_CLIENT — skipping"
fi

# ---------------------------------------------------------------------------
# Verify key imports
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "
import jax; print('jax:', jax.__version__)
import jaxlib; print('jaxlib:', jaxlib.__version__)
import cbfpy; print('cbfpy: ok')
import pybullet; print('pybullet: ok')
import trimesh; print('trimesh:', trimesh.__version__)
import mujoco; print('mujoco:', mujoco.__version__)
import numpy; print('numpy:', numpy.__version__)
import scipy; print('scipy:', scipy.__version__)
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
" 2>&1

# Verify psfjax (imported from repo root)
PYTHONPATH="$REPO_ROOT" python -c "
import psfjax
print('psfjax: ok')
from psfjax.occupancy import OccupancyGrid
from psfjax.safe_region import SafeRegionBuilder
print('psfjax imports: ok')
# Check JAX buffering is available
from psfjax.occupancy import _JAX_AVAILABLE
print('psfjax JAX backend:', _JAX_AVAILABLE)
" 2>&1

echo ""
echo "=== Setup complete ==="
echo "Activate with: source main/.venv/bin/activate"

#!/usr/bin/env bash
# Reproducible ViPE install on thor3 (ssh alias tsf-thor). Run on thor from anywhere:
#   bash setup.sh            # clone + build image tsf-vipe:latest (~15 min, mostly CUDA extension compile)
#   bash setup.sh smoke      # additionally import torch/vipe/vipe_ext on the GPU to verify the build
# Layout: $VIPE_ROOT/{src,cache,out}; weights (torch hub + HF) are cached in $VIPE_ROOT/cache.
set -euo pipefail

VIPE_ROOT=${VIPE_ROOT:-$HOME/tsf/dev/vipe}
VIPE_COMMIT=${VIPE_COMMIT:-8c9f36144e08d8f8c8cf60d20ad70c100d9cff07}
IMAGE=${VIPE_IMAGE:-tsf-vipe:latest}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG=${HOME}/tsf/logs/vipe_setup.log

mkdir -p "$VIPE_ROOT"/{cache,out} "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [ ! -d "$VIPE_ROOT/src/.git" ]; then
    if [ -d "$HOME/tsf/third_party/vipe/.git" ]; then
        git clone --no-hardlinks "$HOME/tsf/third_party/vipe" "$VIPE_ROOT/src"
        git -C "$VIPE_ROOT/src" remote set-url origin https://github.com/nv-tlabs/vipe.git
    else
        git clone https://github.com/nv-tlabs/vipe.git "$VIPE_ROOT/src"
    fi
fi
git -C "$VIPE_ROOT/src" checkout -q "$VIPE_COMMIT"

# Dockerfile and worker live next to this script; the ViPE checkout is the build context.
cp "$HERE/Dockerfile" "$HERE/vipe_worker.py" "$VIPE_ROOT/src/"
docker build -t "$IMAGE" -f "$VIPE_ROOT/src/Dockerfile" "$VIPE_ROOT/src"

if [ "${1:-}" = smoke ]; then
    docker run --rm --runtime nvidia --gpus all --ipc host \
        -v "$VIPE_ROOT/cache:/cache" -v "$VIPE_ROOT/out:/work" "$IMAGE" \
        python -c "import torch, vipe, vipe_ext; print('ok', torch.__version__, torch.cuda.get_device_name(0))"
fi
echo "ViPE image $IMAGE ready"

#!/usr/bin/env bash
set -euo pipefail

IMAGE="tssr-smiles:latest"
ARCH_HINT="linux/amd64"  # change to your build arch; if you built on Intel, keep amd64

# Path to the saved image file (adjust if you put it elsewhere)
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAR_GZ="${PKG_DIR}/tssr-smiles.tar.gz"
TAR_RAW="${PKG_DIR}/tssr-smiles.tar"

have_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1
}

load_image() {
  if [[ -f "$TAR_GZ" ]]; then
    echo ">> Loading Docker image from ${TAR_GZ}..."
    gunzip -c "$TAR_GZ" | docker load
  elif [[ -f "$TAR_RAW" ]]; then
    echo ">> Loading Docker image from ${TAR_RAW}..."
    docker load -i "$TAR_RAW"
  else
    echo "!! Could not find image file (expected ${TAR_GZ} or ${TAR_RAW})." >&2
    echo "   Please download it (see README) and place it next to this script." >&2
    exit 1
  fi
}

# Load image if missing
if ! have_image; then
  load_image
fi

# GPU toggle:
#   - pass --gpu to request GPU (requires NVIDIA drivers + toolkit on host)
#   - pass --cpu to force CPU (sets CUDA_VISIBLE_DEVICES empty)
GPU_FLAG=()
ENV_FLAG=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_FLAG=(--gpus all)
  shift
elif [[ "${1:-}" == "--cpu" ]]; then
  ENV_FLAG=(-e CUDA_VISIBLE_DEVICES=)
  shift
fi

# Profiles: default to "help" if none given.
PROFILE="${1:-help}"

# Run it. ENTRYPOINT in image already launches the profile runner.
exec docker run --rm -it "${GPU_FLAG[@]}" "${ENV_FLAG[@]}" "$IMAGE" "$PROFILE"

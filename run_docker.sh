#!/usr/bin/env bash
set -euo pipefail

# GHCR image published by CI from this repository
IMAGE="ghcr.io/lynaluo-lab/tssr-smiles-generation:latest"

have_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1
}

# Ensure image is available locally; pull if missing.
if ! have_image; then
  echo ">> Pulling ${IMAGE} from GHCR..."
  docker pull "$IMAGE"
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

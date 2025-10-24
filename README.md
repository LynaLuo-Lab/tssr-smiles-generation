# TSSR-SMILES: Reproducible Package for Sequence-Based Molecular Generation with RL

This repository provides a compact, reproduction package for the TSSR-SMILES project. It bundles:
- A pre-built Docker image published to GitHub Container Registry: ghcr.io/lynaluo-lab/tssr-smiles-generation:latest.
- A one-step launcher script run_docker.sh that automatically pulls the image from GHCR (if needed) and runs curated experiment profiles.
- The minimal source files required to execute the training and evaluation pipeline end-to-end on the dataset (under data/).

You do not need to install Python, CUDA, or RDKit locally. Everything runs inside the provided container.


## Contents at a glance
- RunScript.py — main entry point orchestrating pretraining + RL or pure RL.
- RLPipeline/ — core components used by RunScript.py only:
  - CharRNN.py — recurrent generator + critic
  - SequenceDataSet.py — dataset and label encoding
  - SequenceEnv.py — Gym-style environment + reward logic
  - MeanEvaluation.py — evaluation helpers (RDKit-based)
  - BulkGenerator.py — sampling/generation helper
- data/ — MOSES dataset (one SMILES per line). Files used by default:
  - data/train.txt, data/test.txt, data/train.csv, data/test.csv
- Dockerfile, environment.yml — reproducible container build spec (for transparency)
- entrypoint.sh — profile router used by the image
- run_docker.sh — convenience wrapper to pull and run the GHCR image


## Scientific overview
- Problem: sequence-based molecular generation with reinforcement learning (SMILES strings).
- Model: character-level RNN generator trained with either (a) pure RL or (b) finetune + RL.
- Policy optimization: PPO via Tianshou.
- Evaluation: RDKit-driven filters and metrics (see MeanEvaluation.py), plus sampled molecule dumps.
- Reproducibility: fixed seeds baked into predefined profiles; deterministic backend settings where applicable.

This package is intended for qualitative and functional review on MOSES dataset. It demonstrates the training loop, 
reward design, and resulting samples without requiring multi-GPU infrastructure.


## System requirements
- Docker 24+ on Linux, macOS (Apple Silicon via emulation is slower), or Windows 10/11 with WSL2.
- Optional GPU acceleration (recommended):
  - NVIDIA GPU with recent driver and the NVIDIA Container Toolkit (a.k.a. nvidia-docker2) on the host.
  - Verify with: docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
- Disk: ~6–10 GB free for the image and outputs.
- RAM: 8 GB+ (more is better). GPU VRAM: 8 GB+ recommended.


## Quick start (pre-built Docker image)
1) Make the launcher executable
- chmod +x run_docker.sh

2) Show the built-in help and available profiles
- ./run_docker.sh help

3) Run a PRL profile on GPU (fastest if GPU is available)
- ./run_docker.sh --gpu PRL-Run1

4) Run on CPU only (slower, but works everywhere)
- ./run_docker.sh --cpu PRL-Run1

Notes
- The launcher will automatically pull ghcr.io/lynaluo-lab/tssr-smiles-generation:latest if it is not already present locally.
- Profiles use fixed seeds for reproducibility.

## Container image and CI/CD
- This repository is connected to GitHub Container Registry (GHCR). On pushes to main or manual dispatch, GitHub Actions builds the Docker image from the Dockerfile and publishes it.
- Workflow: .github/workflows/publish.yml
- Image names and tags:
  - ghcr.io/lynaluo-lab/tssr-smiles-generation:latest
  - ghcr.io/lynaluo-lab/tssr-smiles-generation:sha-<commit-sha>

## Reproducible experiment profiles
These profile names are routed by entrypoint.sh and ultimately call RunScript.py with fixed random seeds:
- PRL-Run1 … PRL-Run5 — Pure RL (skips Lightning pretraining)
- FRL-Run1 … FRL-Run5 — Finetune with Lightning, then RL

Examples
- Pure RL, third seed: ./run_docker.sh --gpu PRL-Run3
- Finetune+RL, fifth seed: ./run_docker.sh --cpu FRL-Run5

Behind the scenes
- PRL profiles map to: python RunScript.py --pure-rl --seed <fixed>
- FRL profiles map to: python RunScript.py --seed <fixed>


## Persisting outputs beyond the container
By default, outputs are written inside the container to:
- RLPipeline/runs/<mode>/<timestamp>/
This includes TensorBoard logs and sampled molecules.

To persist on the host when using docker run directly, mount a volume
- docker run --rm -it --gpus all 
  -v "\$PWD/data:/workspace/data" 
  -v "\$PWD/outputs:/workspace/RLPipeline/runs" 
  ghcr.io/lynaluo-lab/tssr-smiles-generation:latest PRL-Run1

Alternatively, after a run, you can copy results out of the container with docker cp if you noted the container ID.

The simple run_docker.sh wrapper does not mount host volumes by default to keep usage minimal. For persistent outputs, prefer the manual docker run shown above.


## Manual commands
Pull the image (if you prefer not to use run_docker.sh)
- docker pull ghcr.io/lynaluo-lab/tssr-smiles-generation:latest

List profiles and help
- docker run --rm -it ghcr.io/lynaluo-lab/tssr-smiles-generation:latest help

Run with GPU
- docker run --rm -it --gpus all ghcr.io/lynaluo-lab/tssr-smiles-generation:latest PRL-Run1

Pass raw arguments to RunScript.py (bypass profiles)
- docker run --rm -it --gpus all ghcr.io/lynaluo-lab/tssr-smiles-generation:latest -- --pure-rl --seed 42

Mount custom data and persist outputs together
- docker run --rm -it --gpus all 
  -v "\$PWD/data:/workspace/data" 
  -v "\$PWD/outputs:/workspace/RLPipeline/runs" 
  ghcr.io/lynaluo-lab/tssr-smiles-generation:latest PRL-Run2


## Expected outputs and runtime
- Training progress and metrics are logged to RLPipeline/runs/<mode>/<timestamp>/.
- Generated samples and evaluation summaries are written alongside logs.
- GPU runs typically complete demo-scale profiles in minutes to tens of minutes; CPU runs are substantially slower.


## Reproducibility notes
- Fixed seeds per profile (see entrypoint.sh). For example, FRL-Run1 uses seed 1999133639; PRL-Run1 uses seed 640011233.
- RunScript.py sets deterministic/cuDNN-safe flags where applicable and seeds all major RNGs (Python, NumPy, PyTorch CPU/CUDA, DataLoader workers).
- The Docker image pins the CUDA/PyTorch toolchain (pytorch 2.5.1 + CUDA 12.1 wheels) via environment.yml.


## Troubleshooting
- Permission denied running the script
  - Run: chmod +x run_docker.sh
- Docker "permission denied" without sudo
  - Add your user to the docker group, then re-login; or prefix commands with sudo.
- GPU not detected
  - Check NVIDIA drivers and install the NVIDIA Container Toolkit. Test with: docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
- Cannot pull image from GHCR
  - Check your internet connectivity and that the image name is correct: ghcr.io/lynaluo-lab/tssr-smiles-generation:latest. If the image is private or you encounter rate limits, run: docker login ghcr.io (use a GitHub token with read:packages), then retry.
- Slow training
  - Use a GPU if available, reduce batch sizes, or try a PRL profile for a faster qualitative run.


## Repository layout (for reference)
- RunScript.py — main training/eval driver
- RLPipeline/
  - CharRNN.py, SequenceDataSet.py, SequenceEnv.py, MeanEvaluation.py, BulkGenerator.py
- data/
  - train.txt, test.txt, train.csv, test.csv (demo)
- Dockerfile, environment.yml, entrypoint.sh, run_docker.sh
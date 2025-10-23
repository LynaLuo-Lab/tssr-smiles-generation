Drug-Discovery-Loss-Term — Minimal Review Package

This directory is a self-contained snapshot intended for code review. It includes:

- RunScript.py — the main entry point
- RLPipeline/ — only the modules imported by RunScript.py
  - CharRNN.py (CharRNNModel and Critic)
  - SequenceDataSet.py (SequenceDataset and LabelEncoder)
  - SequenceEnv.py (SequenceEnv and reward logic)
  - MeanEvaluation.py (generation evaluation helpers)
  - BulkGenerator.py (sampling/generation helper)

What’s intentionally excluded
- Checkpoints, logs, notebooks, large data files, and any modules not imported by RunScript.py.

How to use
1) Install dependencies (see requirements.txt). A Conda environment is recommended for RDKit.

   Example with pip (Linux/macOS):
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements.txt

   Note: rdkit is provided via the rdkit-pypi wheel, which may not support all platforms. If it fails, use Conda:
   - conda create -n ddrlt python=3.10
   - conda activate ddrlt
   - conda install -c conda-forge rdkit
   - pip install -r requirements.txt --no-deps

2) Adjust data paths (if you plan to run training):
   RunScript.py and some helpers refer to absolute paths like:
   - /home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/train.txt
   - /home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/test.txt
   - /home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/train.csv
   - /home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/test.csv

   For a quick test or review you can:
   - Replace these with your local paths, or
   - Create a data/ directory at the project root and mirror the same files.

3) Import or run
   - For inspection: open files directly.
   - For import-only smoke test (does not run training):
     python -c "import sys; sys.path.insert(0, 'review_repo'); import RunScript"

   - To run the full script (this will start pretraining and RL if GPUs/deps are available):
     cd review_repo
     python RunScript.py

   CLI arguments:
   - --seed SEED       Optional integer seed. Default: random per run.
   - --pure-rl         Run pure RL (skip Lightning pretraining). Default: finetuned RL.

   Examples:
   - Default (finetune + RL, random seed):
     python RunScript.py
   - Pure RL with random seed:
     python RunScript.py --pure-rl
   - Finetune + RL with fixed seed 1234:
     python RunScript.py --seed 1234
   - Pure RL with fixed seed 42:
     python RunScript.py --pure-rl --seed 42

Notes
- The script uses GPU (if available) via PyTorch Lightning and Tianshou; CPU is supported but slow.
- Training writes outputs to RLPipeline/runs/<mode>/<timestamp>/ inside this review_repo folder.
- MeanEvaluation uses RDKit and related chemistry tooling; ensure those are installed.

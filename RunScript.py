"""
End‑to‑end training/evaluation entry point for SMILES sequence modeling.

This script wires together the dataset, CharRNN model, PPO policy for
reinforcement learning, and evaluation utilities. It focuses on
reproducibility and readability:
- Deterministic seeding across Python/NumPy/PyTorch (CPU+CUDA)
- Clear environment construction for TIanshou collectors
- Modularized generation and mean evaluation reporting

Usage: run the script directly (python RunScript.py) or import main().
"""
from rdkit import RDLogger
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
RDLogger.DisableLog('rdApp.*')
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.tuner import Tuner
from RLPipeline.CharRNN import CharRNNModel, Critic
from RLPipeline.SequenceDataSet import SequenceDataset, LabelEncoder
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from RLPipeline.SequenceEnv import SequenceEnv
from RLPipeline.MeanEvaluation import MeanEvaluator
from RLPipeline.BulkGenerator import Generator
import time
import os
import shutil
import math
import torch.optim as optim
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy.modelfree.ppo import PPOPolicy
from tianshou.utils import TensorboardLogger
from lightning.pytorch.loggers import TensorBoardLogger as tbl
from gymnasium import spaces
import torch.multiprocessing as mp
import random
import numpy as np
# NOTE: set_start_method must be under __main__ guard to avoid recursive spawning in workers
# It will be set inside the main guard below.
torch.set_float32_matmul_precision('medium')


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU/CUDA) and configure deterministic backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Make cuDNN deterministic where applicable and disable TF32 for strict reproducibility
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Top-level worker seeding function for PyTorch DataLoader.
    Uses torch.initial_seed() so it is compatible with spawn and picklable.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

def main(seed_arg: int | None = None):
    """Main entry point that prepares data, trains the model, and evaluates it.

    This function focuses on clarity and reproducibility: it sets up global
    determinism, prepares datasets/dataloaders, constructs the CharRNN model
    and PPO policy, builds Gymnasium environments for sequence generation, and
    finally runs training with logging and periodic evaluation.

    Parameters
    - seed_arg: optional integer seed to fully reproduce a previous run
    """
    # 1) Create a per-run random seed for full reproducibility (overridable via --seed)
    if seed_arg is not None:
        try:
            seed = int(seed_arg)
        except Exception:
            seed = int.from_bytes(os.urandom(4), 'little')
    else:
        seed = int.from_bytes(os.urandom(4), 'little')

    # 2) Configure strict determinism BEFORE any CUDA context is created
    #    Set CuBLAS workspace config and enable deterministic algorithms.
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("PYTHONHASHSEED", "0")
    except Exception:
        pass
    try:
        # warn_only=True avoids hard errors if some ops lack deterministic variants
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # older PyTorch without warn_only
        torch.use_deterministic_algorithms(True)
    # Now seed all RNGs (Python/NumPy/Torch CPU+CUDA) and configure cuDNN
    set_global_seed(seed)
    print(f"[Sanity] Determinism enabled (torch.use_deterministic_algorithms) with CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")

    # DataLoader RNG for shuffling; worker seeding handled by top-level seed_worker()
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    # Limit intra/inter-op threads for more deterministic CPU behavior
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # 2) Datasets and DataLoaders
    # Configure parallel workers via environment variables (sensible defaults for 12C/24T CPU)
    cpu_cnt = os.cpu_count() or 4
    DL_WORKERS = int(os.getenv("DL_WORKERS", str(max(2, min(8, cpu_cnt // 2)))))
    pin_mem = torch.cuda.is_available()
    prefetch_kwargs = {"prefetch_factor": 2} if DL_WORKERS > 0 else {}
    print(f"[Sanity] DataLoader workers: {DL_WORKERS} (pin_memory={pin_mem})")

    train_path = '/home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/train.txt'
    test_path = '/home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/test.txt'
    dataset_train = SequenceDataset(train_path)
    dataset_test = SequenceDataset(test_path)
    train_loader = DataLoader(
        dataset_train,
        batch_size=512,
        shuffle=True,
        num_workers=DL_WORKERS,
        persistent_workers=(DL_WORKERS > 0),
        pin_memory=pin_mem,
        worker_init_fn=seed_worker,
        generator=dl_gen,
        **prefetch_kwargs,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=512,
        shuffle=False,
        num_workers=DL_WORKERS,
        persistent_workers=(DL_WORKERS > 0),
        pin_memory=pin_mem,
        worker_init_fn=seed_worker,
        generator=dl_gen,
        **prefetch_kwargs,
    )
    original_smiles = []
    with open('/home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/train.csv', 'r') as f:
        for line in f.readlines():
            original_smiles.append(line.strip())
    original_smiles_set = set(original_smiles)

    original_smiles_validation = []
    with open('/home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/test.csv', 'r') as f:
        for line in f.readlines():
            original_smiles_validation.append(line.strip())


    encoder = LabelEncoder()
    # Config: choose training mode and LRs
    PURE_RL = True  # True: pure RL (no Lightning pretrain). False: finetuned RL (pretrain with Lightning first)
    USE_PPO_SCHEDULER = False  # If True, enable warmup+cosine LR scheduler for PPO optimizer
    PRETRAIN_LR = 1e-4  # LR used only when finetuning with Lightning
    RL_LR_PURE = 1e-4   # PPO optimizer LR for pure RL
    RL_LR_FINETUNED = 1e-8  # PPO optimizer LR for finetuned RL

    print("[Sanity] Mode:", "Running PRL (Pure RL)" if PURE_RL else "Running FT-RL (Finetune + RL)")

    # Hyperparameters
    layer_count = 3
    pad_idx =  encoder.cti['[PAD]']
    hidden_size = 512
    dropout = 0.2
    lr = PRETRAIN_LR

    # Per-run output directory (separated by method)
    method_name = "pure_rl" if PURE_RL else "finetuned_rl"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("RLPipeline", "runs", method_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "run_stats.txt")
    # Record the seed to console and stats file for exact reproducibility
    print(f"[Sanity] Random seed for this run: {seed}")
    with open(stats_path, "a") as sf:
        sf.write(f"Random Seed: {seed}\n")
        sf.write(f"DataLoader workers: {DL_WORKERS} (pin_memory={pin_mem})\n")
        sf.write(f"Deterministic torch: True\n")
        sf.write(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}\n")
        try:
            sf.write(f"torch_num_threads: {torch.get_num_threads()}, torch_num_interop_threads: {torch.get_num_interop_threads()}\n")
        except Exception:
            pass

    max_epochs = 1

    embed_dim = dataset_train.vocab_size * 2

    model = CharRNNModel(dataset_test.vocab_size, layer_count,
                         pad_idx, lr, hidden_size=hidden_size,
                         dropout=dropout,embedding_dim=embed_dim )

    if PURE_RL:
        # Pure RL: skip Lightning pretraining
        print("[Sanity] Skipping pretraining (PURE RL mode)")
        ckpt = None
        pretrained = model
    else:
        print("[Sanity] Starting FT pretraining with Lightning...")
        ckpt_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath='checkpointsCharRNN/',
                                filename="best-{epoch:02d}-{val_loss:.2f}")
        trainer = Trainer(
            default_root_dir="RLPipeline/lr_find_ckpts",
            max_epochs=max_epochs,
            accelerator="cuda",
            precision="32-true",
            gradient_clip_val=10.0,
            deterministic=True,
            logger=tbl("RLPipeline/tb_logs", name="char_rnn"),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=1),
                RichProgressBar(),
                ckpt_callback,
            ],
        )
        tuner = Tuner(trainer)
        tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
            min_lr=1e-6,
            max_lr=1.0,
            early_stop_threshold=None,
            update_attr=True,
        )
        trainer.fit(model, train_loader, test_loader)
        ckpt = ckpt_callback.best_model_path
        print(f"[Sanity] Pretraining complete. Best checkpoint: {ckpt}")
        pretrained = CharRNNModel.load_from_checkpoint(ckpt)
    pretrained.eval()
    # Move model to device before generation for speed while preserving determinism
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = pretrained.to(DEVICE)

    # Hyperparams
    sample_size = 10_000

    start = time.perf_counter()

    vocab = encoder.characters
    eos_id = vocab.index('[EOS]')
    bos_id = vocab.index('[BOS]')
    gen = Generator(pretrained, bos_id, eos_id, 100)
    print(f"[Sanity] Pre-RL: generating and evaluating {sample_size} samples...")
    means_eval = MeanEvaluator(gen, vocab, eos_id, sample_size)
    means = means_eval.get_means()
    print("[Sanity] Pre-RL evaluation complete.")

    path_valid = os.path.join(run_dir, "before_samples_valid")
    path_fixed = os.path.join(run_dir, "before_samples_fixed")
    path_unparsable_to_valid = os.path.join(run_dir, "before_unparsable_to_valid.txt")
    path_initial_to_fully_fixed = os.path.join(run_dir, "before_initial_to_fully_fixed.txt")

    means_eval.write_samples_to_file(path_valid, path_fixed, path_unparsable_to_valid, path_initial_to_fully_fixed)

    # Summarize and record sample caches (BEFORE RL)
    try:
        nv_count = len(means_eval.eval_calc.near_valid_cache)
        fixed_before_count = len(means_eval.eval_calc.fixed_cache_before)
        fixed_after_count = len(means_eval.eval_calc.fixed_cache_after)
        unparse_to_valid_count = len(means_eval.eval_calc.unparsable_to_valid_pairs)
        initial_to_fixed_count = len(means_eval.eval_calc.initial_to_fully_fixed_pairs)
        print(f"[MeanEval][Before] near_valid={nv_count}, fixed_before={fixed_before_count}, fixed_after={fixed_after_count}, unparsable→valid_pairs={unparse_to_valid_count}, initial→fully_fixed_pairs={initial_to_fixed_count}")
        with open(stats_path, "a") as sf:
            sf.write("=== SAMPLES (BEFORE RL) ===\n")
            sf.write(f"near_valid_count: {nv_count}\n")
            sf.write(f"fixed_before_count: {fixed_before_count}\n")
            sf.write(f"fixed_after_count: {fixed_after_count}\n")
            sf.write(f"unparsable_to_valid_pairs: {unparse_to_valid_count}\n")
            sf.write(f"initial_to_fully_fixed_pairs: {initial_to_fixed_count}\n")
            sf.write(f"near_valid_file: {path_valid}.txt\n")
            sf.write(f"fixed_before_file: {path_fixed}.txt\n")
            sf.write(f"fixed_after_file: {path_fixed}_fixed.txt\n")
            sf.write(f"unparsable_to_valid_file: {path_unparsable_to_valid}\n")
            sf.write(f"initial_to_fully_fixed_file: {path_initial_to_fully_fixed}\n")
    except Exception as e:
        print(f"[Warning] Failed to summarize BEFORE RL sample caches: {e}")
    stats = means_eval.eval_valid_novel(original_smiles_set)
    before_means_str = (
        f"Average Length: {means[0]}, "
        f"Average Swaps: {means[2]}, "
        f"Average Fixed: {means[3]}, "
        f"Average Err_change: {means[4]}\n"
    )
    before_stats_str = (
        f"Syntactically Valid molecules: {stats[0]}, "
        f"Novel molecules: {stats[1]}, "
        f"Chemically Valid Molecules: {stats[2]}, "
        f"Percentage of Syntactically Valid: {stats[3] * 100:.4f}%, "
        f"Percentage of Novel: {stats[4] * 100:.4f}%, "
        f"Percentage of Chemically Valid: {stats[5] * 100:.4f}%, "
        f"Chemically Valid Similarity (Scaffold): {stats[6]:.4f}\n"
    )
    print(before_means_str)
    print(before_stats_str)
    with open(stats_path, "a") as sf:
        sf.write("=== BEFORE RL ===\n")
        sf.write(before_means_str)
        sf.write(before_stats_str)

    # Save models/checkpoints BEFORE RL
    try:
        torch.save(pretrained.state_dict(), os.path.join(run_dir, "before_rl_actor.pth"))
        print(f"[Sanity] Saved pre-RL actor -> {os.path.join(run_dir, 'before_rl_actor.pth')}")
    except Exception as e:
        print(f"[Warning] Failed to save before_rl_actor.pth: {e}")
    if not PURE_RL and ckpt:
        try:
            shutil.copy2(ckpt, os.path.join(run_dir, "pretrain_best.ckpt"))
            print(f"[Sanity] Copied pretrain best ckpt -> {os.path.join(run_dir, 'pretrain_best.ckpt')}")
        except Exception as e:
            print(f"[Warning] Failed to copy pretrain best checkpoint: {e}")

    # Write all chemically valid molecules (including fixed) to CSV (BEFORE RL)
    try:
        before_valid_csv = os.path.join(run_dir, "before_valid_all.csv")
        means_eval.write_all_valid_to_csv(before_valid_csv, include_fixed=True)
        # Also write a plain list (TXT) of all valid molecules derived from pair RHS (unparsable→valid and initial→fully_fixed)
        before_all_valid_txt = os.path.join(run_dir, "before_all_valid.txt")
        before_all_valid_count = means_eval.write_valid_pairs_list(before_all_valid_txt)
        # Compute additional metrics and record
        rows_before = means_eval.collect_chemically_valid(include_fixed=True)
        valid_smiles_before = [s for s, _ in rows_before]
        metrics_before = means_eval.compute_additional_metrics(valid_smiles_before)
        means_eval.write_metrics_csv(os.path.join(run_dir, "before_metrics.csv"), metrics_before)
        # Record the valid CSV path and count in stats
        with open(stats_path, "a") as sf:
            sf.write("=== VALID MOLECULES (BEFORE RL) ===\n")
            sf.write(f"valid_csv: {before_valid_csv}\n")
            sf.write(f"valid_count_rows: {len(rows_before)}\n")
            sf.write(f"all_valid_pairs_file: {before_all_valid_txt}\n")
            sf.write(f"all_valid_pairs_count: {before_all_valid_count}\n")
        means_eval.append_metrics_to_stats(stats_path, "ADDITIONAL METRICS (BEFORE RL)", metrics_before)
    except Exception as e:
        print(f"[Warning] Failed to compute/write BEFORE RL valid CSV/metrics: {e}")

    end = time.perf_counter()

    ROLLOUT_STEPS = 512  # per epoch across all envs
    EPOCHS = 1000
    BATCH_SIZE = 512
    REPEAT_PER_COLLECT = 1
    STEP_PER_COLLECT = 60  # number of env steps per data collection in Tianshou trainer
    RL_LR = RL_LR_PURE if PURE_RL else RL_LR_FINETUNED
    GAMMAa = 0.99
    GAE_LAMBDA = 0.95
    EPS_CLIP = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ENV Hyperparams
    LATENT_SIZE = None
    K_SUBST = 8
    ALPHA = 0.2
    BETA = 0.5
    GAMMA = 0.3

    def make_env(seed: int):
        def _init():
            # Do NOT reset global RNGs here when using DummyVectorEnv; main process already seeded.
            # SequenceEnv.seed(seed) should configure any internal RNGs used by the environment.
            env = SequenceEnv(
                vocab=vocab,
                max_len=60,
                means=means,
                latent_dim=LATENT_SIZE,
                k_subst=K_SUBST,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )
            env.seed(seed)
            return env

        return _init
    # 3) Vectorized environments
    cpu_cnt = os.cpu_count() or 4
    RL_ENVS = 1 ; RL_TEST_ENVS = 1

    # Use DummyVectorEnv when using a single environment to avoid subprocess nondeterminism
    if RL_ENVS == 1:
        train_envs = DummyVectorEnv([make_env(seed)])
    else:
        train_envs = SubprocVectorEnv([make_env(seed + i) for i in range(RL_ENVS)])

    if RL_TEST_ENVS == 1:
        test_envs = DummyVectorEnv([make_env(seed + 1000)])
    else:
        test_envs = SubprocVectorEnv([make_env(seed + 1000 + i) for i in range(RL_TEST_ENVS)])

    actor = pretrained.to(DEVICE)
    critic = Critic(actor)
    optimizer = optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=RL_LR)

    # Optional PPO LR scheduler: warmup + cosine with floor, parameterized by optimizer update steps
    scheduler = None
    if USE_PPO_SCHEDULER:
        # Derive expected number of gradient updates in Tianshou
        collects_per_epoch = math.ceil(ROLLOUT_STEPS / STEP_PER_COLLECT)
        updates_per_collect = max(1, math.ceil(STEP_PER_COLLECT / BATCH_SIZE))
        updates_per_epoch = collects_per_epoch * updates_per_collect * REPEAT_PER_COLLECT
        TOTAL_UPDATES = EPOCHS * updates_per_epoch

        warmup_updates = max(1, int(0.05 * TOTAL_UPDATES))  # 5% warmup
        eta_min = max(1e-5, RL_LR * 0.1)

        warmup = LambdaLR(optimizer, lr_lambda=lambda s: min((s + 1) / warmup_updates, 1.0))
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, TOTAL_UPDATES - warmup_updates), eta_min=eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_updates])

        print(f"[Sanity] PPO LR Scheduler: ENABLED (updates total={TOTAL_UPDATES}, warmup_updates={warmup_updates}, eta_min={eta_min})")
        with open(stats_path, "a") as sf:
            sf.write(f"PPO Scheduler: Enabled (total_updates={TOTAL_UPDATES}, warmup_updates={warmup_updates}, eta_min={eta_min})\n")
    else:
        print("[Sanity] PPO LR Scheduler: DISABLED")
        with open(stats_path, "a") as sf:
            sf.write("PPO Scheduler: Disabled\n")

    dist_fn = lambda logits: torch.distributions.Categorical(logits=logits)
    observation_space = spaces.Discrete(len(vocab))
    action_space = spaces.Discrete(len(vocab))
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=dist_fn,
        action_space=action_space,
        observation_space=observation_space,
        eps_clip=EPS_CLIP,
        value_clip=True,
        advantage_normalization=False,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        gae_lambda=GAE_LAMBDA,
        discount_factor=GAMMAa,
        reward_normalization=False,
        deterministic_eval=False,
        action_scaling=False,
        lr_scheduler=scheduler,
    ).to(DEVICE)
    train_collector = Collector(policy, train_envs)
    test_collector = Collector(policy, test_envs)

    logdir = os.path.join(run_dir, "tb")
    writer = SummaryWriter(log_dir=logdir)
    logger = TensorboardLogger(writer)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCHS,
        step_per_epoch=ROLLOUT_STEPS,
        repeat_per_collect=REPEAT_PER_COLLECT,
        step_per_collect=STEP_PER_COLLECT,
        episode_per_test=10,
        batch_size=BATCH_SIZE,
        logger=logger,
    )
    print("[Sanity] Starting PPO training...")
    result = trainer.run()
    print("[Sanity] PPO training complete.")
    print("Best reward:",        result.best_reward)
    print("Best reward std:",    result.best_reward_std)
    print("Gradient steps:",     result.gradient_step)
    print("Total train steps:",  result.train_step)

    # Save PPO artifacts and trained models AFTER RL
    try:
        torch.save(policy.actor.state_dict(), os.path.join(run_dir, "after_rl_actor.pth"))
        torch.save(policy.critic.state_dict(), os.path.join(run_dir, "after_rl_critic.pth"))
        torch.save(policy.state_dict(), os.path.join(run_dir, "ppo_sequence_policy.pth"))
        print(f"[Sanity] Saved PPO artifacts -> {run_dir}")
    except Exception as e:
        print(f"[Warning] Failed to save PPO artifacts: {e}")

    # Write RL training summary to stats file
    with open(stats_path, "a") as sf:
        sf.write("=== RL TRAINING SUMMARY ===\n")
        sf.write(f"Best reward: {result.best_reward}\n")
        sf.write(f"Best reward std: {result.best_reward_std}\n")
        sf.write(f"Gradient steps: {result.gradient_step}\n")
        sf.write(f"Total train steps: {result.train_step}\n")

    start = time.perf_counter()

    # Before heavy post-RL evaluation, clear any residual CUDA allocations to reduce
    # peak memory during RDKit processing and large CPU arrays.
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("[Sanity] Cleared CUDA cache before post-RL evaluation.")
    except Exception:
        pass

    vocab = encoder.characters
    eos_id = vocab.index('[EOS]')
    bos_id = vocab.index('[BOS]')
    gen = Generator(pretrained,bos_id,eos_id, 100)
    means_eval = MeanEvaluator(gen, vocab, eos_id, sample_size)
    means_after = means_eval.get_means()

    path_valid = os.path.join(run_dir, "after_samples_valid")
    path_fixed = os.path.join(run_dir, "after_samples_fixed")
    path_unparsable_to_valid = os.path.join(run_dir, "after_unparsable_to_valid.txt")
    path_initial_to_fully_fixed = os.path.join(run_dir, "after_initial_to_fully_fixed.txt")

    means_eval.write_samples_to_file(path_valid, path_fixed, path_unparsable_to_valid, path_initial_to_fully_fixed)

    # Summarize and record sample caches (AFTER RL)
    try:
        nv_count_a = len(means_eval.eval_calc.near_valid_cache)
        fixed_before_count_a = len(means_eval.eval_calc.fixed_cache_before)
        fixed_after_count_a = len(means_eval.eval_calc.fixed_cache_after)
        unparse_to_valid_count_a = len(means_eval.eval_calc.unparsable_to_valid_pairs)
        initial_to_fixed_count_a = len(means_eval.eval_calc.initial_to_fully_fixed_pairs)
        print(f"[MeanEval][After] near_valid={nv_count_a}, fixed_before={fixed_before_count_a}, fixed_after={fixed_after_count_a}, unparsable→valid_pairs={unparse_to_valid_count_a}, initial→fully_fixed_pairs={initial_to_fixed_count_a}")
        with open(stats_path, "a") as sf:
            sf.write("=== SAMPLES (AFTER RL) ===\n")
            sf.write(f"near_valid_count: {nv_count_a}\n")
            sf.write(f"fixed_before_count: {fixed_before_count_a}\n")
            sf.write(f"fixed_after_count: {fixed_after_count_a}\n")
            sf.write(f"unparsable_to_valid_pairs: {unparse_to_valid_count_a}\n")
            sf.write(f"initial_to_fully_fixed_pairs: {initial_to_fixed_count_a}\n")
            sf.write(f"near_valid_file: {path_valid}.txt\n")
            sf.write(f"fixed_before_file: {path_fixed}.txt\n")
            sf.write(f"fixed_after_file: {path_fixed}_fixed.txt\n")
            sf.write(f"unparsable_to_valid_file: {path_unparsable_to_valid}\n")
            sf.write(f"initial_to_fully_fixed_file: {path_initial_to_fully_fixed}\n")
    except Exception as e:
        print(f"[Warning] Failed to summarize AFTER RL sample caches: {e}")

    stats_after = means_eval.eval_valid_novel(original_smiles_set)
    print(
        f"Average Length Before: {means[0]}, "
        f"Average Swaps Before: {means[2]}, "
        f"Average Fixed Before: {means[3]}, "
        f"Average Err_change Before: {means[4]}\n"
    )

    print(
        f"Average Length After: {means_after[0]}, "
        f"Average Swaps After: {means_after[2]}, "
        f"Average Fixed After: {means_after[3]}, "
        f"Average Err_change After: {means_after[4]}\n"
    )
    print(
        f"Delta Average Length (After - Before): {means_after[0] - means[0]}\n"
    )

    print(
        f"Syntactically Valid molecules Before: {stats[0]}, "
        f"Novel molecules Before: {stats[1]}, "
        f"Chemically Valid Molecules Before: {stats[2]}, "
        f"Percentage of Syntactically Valid Before: {stats[3] * 100:.4f}%, "
        f"Percentage of Novel Before: {stats[4] * 100:.4f}%, "
        f"Percentage of Chemically Valid Before: {stats[5] * 100:.4f}%, "
        f"Chemically Valid Similarity (Scaffold) Before: {stats[6]:.4f}\n"
    )

    print(
        f"Syntactically Valid molecules After: {stats_after[0]}, "
        f"Novel molecules After: {stats_after[1]}, "
        f"Chemically Valid Molecules After: {stats_after[2]}, "
        f"Percentage of Syntactically Valid After: {stats_after[3] * 100:.4f}%, "
        f"Percentage of Novel After: {stats_after[4] * 100:.4f}%, "
        f"Percentage of Chemically Valid After: {stats_after[5] * 100:.4f}%, "
        f"Chemically Valid Similarity (Scaffold) After: {stats_after[6]:.4f}\n"
    )

    # Write AFTER stats to stats file
    after_means_before_str = (
        f"Average Length Before: {means[0]}, "
        f"Average Swaps Before: {means[2]}, "
        f"Average Fixed Before: {means[3]}, "
        f"Average Err_change Before: {means[4]}\n"
    )
    after_means_after_str = (
        f"Average Length After: {means_after[0]}, "
        f"Average Swaps After: {means_after[2]}, "
        f"Average Fixed After: {means_after[3]}, "
        f"Average Err_change After: {means_after[4]}\n"
    )
    after_stats_before_str = (
        f"Syntactically Valid molecules Before: {stats[0]}, "
        f"Novel molecules Before: {stats[1]}, "
        f"Chemically Valid Molecules Before: {stats[2]}, "
        f"Percentage of Syntactically Valid Before: {stats[3] * 100:.4f}%, "
        f"Percentage of Novel Before: {stats[4] * 100:.4f}%, "
        f"Percentage of Chemically Valid Before: {stats[5] * 100:.4f}%, "
        f"Chemically Valid Similarity (Scaffold) Before: {stats[6]:.4f}\n"
    )
    after_stats_after_str = (
        f"Syntactically Valid molecules After: {stats_after[0]}, "
        f"Novel molecules After: {stats_after[1]}, "
        f"Chemically Valid Molecules After: {stats_after[2]}, "
        f"Percentage of Syntactically Valid After: {stats_after[3] * 100:.4f}%, "
        f"Percentage of Novel After: {stats_after[4] * 100:.4f}%, "
        f"Percentage of Chemically Valid After: {stats_after[5] * 100:.4f}%, "
        f"Chemically Valid Similarity (Scaffold) After: {stats_after[6]:.4f}\n"
    )
    with open(stats_path, "a") as sf:
        sf.write("=== AFTER RL ===\n")
        sf.write(after_means_before_str)
        sf.write(after_means_after_str)
        # Write delta for Average Length
        sf.write(f"Delta Average Length (After - Before): {means_after[0] - means[0]}\n")
        sf.write(after_stats_before_str)
        sf.write(after_stats_after_str)

    # Write all chemically valid molecules and additional metrics (AFTER RL)
    try:
        after_valid_csv = os.path.join(run_dir, "after_valid_all.csv")
        means_eval.write_all_valid_to_csv(after_valid_csv, include_fixed=True)
        # Also write a plain list (TXT) of all valid molecules derived from pair RHS (unparsable→valid and initial→fully_fixed)
        after_all_valid_txt = os.path.join(run_dir, "after_all_valid.txt")
        after_all_valid_count = means_eval.write_valid_pairs_list(after_all_valid_txt)
        rows_after = means_eval.collect_chemically_valid(include_fixed=True)
        valid_smiles_after = [s for s, _ in rows_after]
        metrics_after = means_eval.compute_additional_metrics(valid_smiles_after)
        means_eval.write_metrics_csv(os.path.join(run_dir, "after_metrics.csv"), metrics_after)
        # Record the valid CSV path and count in stats
        with open(stats_path, "a") as sf:
            sf.write("=== VALID MOLECULES (AFTER RL) ===\n")
            sf.write(f"valid_csv: {after_valid_csv}\n")
            sf.write(f"valid_count_rows: {len(rows_after)}\n")
            sf.write(f"all_valid_pairs_file: {after_all_valid_txt}\n")
            sf.write(f"all_valid_pairs_count: {after_all_valid_count}\n")
        means_eval.append_metrics_to_stats(stats_path, "ADDITIONAL METRICS (AFTER RL)", metrics_after)
    except Exception as e:
        print(f"[Warning] Failed to compute/write AFTER RL valid CSV/metrics: {e}")

    end = time.perf_counter()



if __name__ == '__main__':
    import argparse
    # Ensure safe multiprocessing start method only when running as the main module
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set; ignore to avoid crashes
        pass
    parser = argparse.ArgumentParser(description='Run RL training/evaluation with optional fixed seed')
    parser.add_argument('--seed', type=int, default=None, help='Random seed to use for this run')
    args = parser.parse_args()
    main(args.seed)

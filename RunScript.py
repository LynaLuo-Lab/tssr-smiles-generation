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
import argparse
# NOTE: set_start_method must be under __main__ guard to avoid recursive spawning in workers
# It will be set inside the main guard below.
torch.set_float32_matmul_precision('medium')


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU/CUDA) and configure deterministic backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic where applicable
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

def main():
    parser = argparse.ArgumentParser(description="Run training/eval for sequence RL.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs. Default: random per run.")
    parser.add_argument("--pure-rl", dest="pure_rl", action="store_true", help="Run pure RL (skip Lightning pretraining).")
    parser.set_defaults(pure_rl=False)
    args = parser.parse_args()

    # 1) Create and set a per-run random seed for full reproducibility
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
    set_global_seed(seed)

    # DataLoader RNG for shuffling; worker seeding handled by top-level seed_worker()
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    # 2) Datasets and DataLoaders
    train_path = 'data/train.txt'
    test_path = 'data/test.txt'
    dataset_train = SequenceDataset(train_path)
    dataset_test = SequenceDataset(test_path)
    train_loader = DataLoader(dataset_train, batch_size=512, shuffle=True, num_workers=2,
                              worker_init_fn=seed_worker, generator=dl_gen)
    test_loader = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=2,
                             worker_init_fn=seed_worker, generator=dl_gen)
    original_smiles = []
    with open('data/train.csv', 'r') as f:
        for line in f.readlines():
            original_smiles.append(line.strip())
    original_smiles_set = set(original_smiles)

    original_smiles_validation = []
    with open('data/test.csv', 'r') as f:
        for line in f.readlines():
            original_smiles_validation.append(line.strip())


    encoder = LabelEncoder()
    # Config: choose training mode and LRs
    PURE_RL = args.pure_rl  # True: pure RL (no Lightning pretrain). False: finetuned RL (pretrain with Lightning first)
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
    run_dir = os.path.join("runs", method_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "run_stats.txt")
    # Record the seed to console and stats file for exact reproducibility
    print(f"[Sanity] Random seed for this run: {seed}")
    with open(stats_path, "a") as sf:
        sf.write(f"Random Seed: {seed}\n")
        sf.write(f"Mode: {'PURE_RL' if PURE_RL else 'FINETUNED_RL'}\n")

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
            default_root_dir="lr_find_ckpts",
            max_epochs=max_epochs,
            accelerator="cuda",
            precision="16-mixed",
            gradient_clip_val=10.0,
            logger=tbl("tb_logs", name="char_rnn"),
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

    # Hyperparams
    sample_size = 10_000

    start = time.perf_counter()

    vocab = encoder.characters
    eos_id = vocab.index('[EOS]')
    bos_id = vocab.index('[BOS]')
    gen = Generator(pretrained,bos_id,eos_id, 100)
    print(f"[Sanity] Pre-RL: generating and evaluating {sample_size} samples...")
    means_eval = MeanEvaluator(gen, vocab, eos_id, sample_size)
    means = means_eval.get_means()
    print("[Sanity] Pre-RL evaluation complete.")

    path_valid = os.path.join(run_dir, "before_samples_valid")
    path_fixed = os.path.join(run_dir, "before_samples_fixed")
    path_unparsable_to_valid = os.path.join(run_dir, "before_unparsable_to_valid.txt")
    path_initial_to_fully_fixed = os.path.join(run_dir, "before_initial_to_fully_fixed.txt")

    means_eval.write_samples_to_file(path_valid, path_fixed, 200, path_unparsable_to_valid, path_initial_to_fully_fixed)
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
    train_envs = DummyVectorEnv([make_env(seed + i) for i in range(1)])
    test_envs = DummyVectorEnv([make_env(seed + i) for i in range(1)])

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
    print("Timing (s):",         result.timing.total_time)
    torch.save(policy.state_dict(), "ppo_sequence_policy.pth")

    # Write RL training summary to stats file
    with open(stats_path, "a") as sf:
        sf.write("=== RL TRAINING SUMMARY ===\n")
        sf.write(f"Best reward: {result.best_reward}\n")
        sf.write(f"Best reward std: {result.best_reward_std}\n")
        sf.write(f"Gradient steps: {result.gradient_step}\n")
        sf.write(f"Total train steps: {result.train_step}\n")
        sf.write(f"Timing (s): {result.timing.total_time}\n")

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

    means_eval.write_samples_to_file(path_valid, path_fixed, 200, path_unparsable_to_valid, path_initial_to_fully_fixed)


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

    end = time.perf_counter()



if __name__ == '__main__':
    # Ensure safe multiprocessing start method only when running as the main module
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set; ignore to avoid crashes
        pass
    main()

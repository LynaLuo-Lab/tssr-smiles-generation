import math
import random
from collections import deque, Counter
from typing import Any

from gymnasium.utils import seeding
from sympy import false

from .SequenceDataSet import LabelEncoder
import numpy as np, torch
from gymnasium import spaces, Env
from rdkit import Chem



def clipped_power_swaps(s, N, m):
    if s >= N:
        return 0.0

    # compute shape parameter
    if m is None or m <= 0 or m >= N:
        gamma = 1.0
    else:
        gamma = math.log(0.5) / math.log(1 - m / N)

    # power‐law decay
    return (1 - s / N) ** gamma

def build_token_freq() -> np.ndarray:
    """
    Count every token in `path` except the three specials.
    Returns freq[i] aligned with encoder.characters, dtype=float, sum = 1.
    """
    encoder = LabelEncoder()
    counter = Counter()
    specials = {"[BOS]", "[EOS]", "[PAD]"}

    with open('data/train.txt', "r") as f:
        for line in f:
            for tok in encoder.tokenize(line.strip()):
                if tok not in specials:
                    counter[tok] += 1

    # map counts onto a vector the size of the vocab
    freq = np.zeros(encoder.vocab_size, dtype=float)
    for tok, cnt in counter.items():
        freq[encoder.cti[tok]] = cnt

    total = freq.sum()
    if total == 0:
        raise ValueError("No non-special tokens counted!")
    return freq / total


class SequenceEnv(Env):
    def __init__(self, vocab: list, max_len: int, means: tuple, latent_dim: int = None,
                 k_subst: int = 8, alpha: float = .2, beta: float = .5, gamma: float = .3):

        self.max_length = max_len
        self.vocab = vocab
        self.eos_id = vocab.index('[EOS]')
        self.bos_id = vocab.index('[BOS]')
        self.vocab_size = len(self.vocab)
        self.means = means
        self.observation_space = spaces.Discrete(len(self.vocab))
        self.action_space = spaces.Discrete(self.vocab_size)
        self.k_subst = k_subst
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.two_stage_reward = TwoStageReward(vocab=self.vocab, k_subst=self.k_subst,
                                               alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        self.np_random, _ = seeding.np_random(None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seq = []
        obs = np.int64(self.bos_id)
        return obs, {}

    def step(self, action):
        action = int(action)
        self.seq.append(action)
        terminated = action == self.eos_id
        truncated = len(self.seq) >= self.max_length
        done = terminated or truncated
        reward = self._reward_fn(self.seq) if done else 0.0
        obs = np.int64(action)
        # Important: return the correct truncated flag so the collector can reset properly
        return obs, reward, terminated, truncated, {}

    def render(self):
        return "".join(self.vocab[i] for i in self.seq)

    def close(self):
        pass

    def _reward_fn(self, sequence):
        smiles = [self.vocab[i] for i in sequence]
        swaps_reward = self.two_stage_reward(smiles[:-1])
        return swaps_reward

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)  # official helper
        # make every RNG in your stack deterministic too
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TwoStageReward:

    def __init__(self,
                 vocab: list[str],
                 store_threshold: int = 3,
                 cache_size: int = 200,
                 k_subst: int = 8,
                 alpha: float = 0.3,
                 beta: float = 0.5,
                 gamma: float = 0.2,):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vocab = vocab
        self.probs = build_token_freq()
        self.k_subst = k_subst
        self.store_thr = store_threshold
        self.near_valid_cache = deque(maxlen=cache_size)

    def __call__(self, smiles: list) -> float:
        """Compute reward and maybe cache the molecule."""
        initially_working, fixed, mol = self._syntax_and_errors(smiles)

        if initially_working:
            failed_swaps, err_diff, distance_from_no_problems = self._try_reduce_chem_problems(mol)
            failed_swaps_term = 1.0/(1 + failed_swaps)
            err_diff_term = err_diff
            return self.alpha * failed_swaps_term + self.beta * err_diff_term + self.gamma * distance_from_no_problems
        elif fixed:
            failed_swaps, err_diff, distance_from_no_problems = self._try_reduce_chem_problems(mol)
            failed_swaps_term = 1.0/(1 + failed_swaps)
            err_diff_term = err_diff
            invalid_penalty = -0.5
            return -0.5 + self.alpha * failed_swaps_term + self.beta * err_diff_term + self.gamma * distance_from_no_problems
        else:
            return -1.0



    def _syntax_and_errors(self, smi: list) -> tuple[bool, bool, Any | None]:
        """
        Returns (syntax_parses?, n_chem_errors)
        Performs swaps if the initial string fails to parse.
        """
        mol = ''.join(smi)
        m = Chem.MolFromSmiles(mol, sanitize=False)
        initially_working = True
        fixed = False
        if m is None:
            initially_working = False
            smil = self._try_syntax_fix(smi)
            if smil is None:
                return initially_working, fixed, None
            else:
                fixed = True
                return initially_working, fixed, smil
        else:
            return initially_working, fixed, smi

    def _try_syntax_fix(self, smi: list) -> list | None:
        """
        Attempt to repair syntax with character substitutions.
        Returns the new SMILES if parsing succeeds, else None.
        """
        chars = smi.copy()
        for pos in random.sample(range(len(chars)), len(chars)):
            cands = self._sample_unique_tokens()
            original = chars[pos]
            for new_c in cands:
                if new_c == original:
                    continue
                chars[pos] = new_c
                if Chem.MolFromSmiles(''.join(chars), sanitize=False) is not None:
                    return chars
            chars[pos] = original
        return None

    def _sample_unique_tokens(self) -> list[str]:
        """k-substitutes, weighted by dataset frequency, no duplicates."""
        idxs = np.random.choice(len(self.vocab),
                                size=min(self.k_subst, (self.probs > 0).sum()),
                                replace=False,
                                p=self.probs)
        return [self.vocab[i] for i in idxs]

    def _try_reduce_chem_problems(self, smi: list) -> tuple[int, float, float]:
        chars = smi.copy()
        best_mol = Chem.MolFromSmiles(''.join(chars), sanitize=False)
        best_err = len(Chem.DetectChemistryProblems(best_mol))
        initial_err = len(Chem.DetectChemistryProblems(best_mol))

        if 0 < initial_err <= self.store_thr:
            self.near_valid_cache.append(best_mol)

        fail_swaps = 0
        if best_err == 0:
            return 0, 0 , 1.0

        for pos in random.sample(range(len(chars)), len(chars)):
            cands = self._sample_unique_tokens()
            original = chars[pos]
            for new_c in cands:
                if new_c == original:
                    continue

                chars[pos] = new_c
                cand_mol = Chem.MolFromSmiles(''.join(chars), sanitize=False)
                if cand_mol is None:
                    chars[pos] = original
                    fail_swaps += 1
                    continue

                n_err = len(Chem.DetectChemistryProblems(cand_mol))
                if n_err < best_err:
                    best_err = n_err
                    best_mol = cand_mol
                    if best_err == 0:
                        return fail_swaps, (initial_err - best_err)/initial_err, 1.0 - (best_err / 12)
                    break
                else:
                    chars[pos] = original
                    fail_swaps += 1
        return fail_swaps, (initial_err - best_err)/initial_err, 1.0 - (best_err / 12)

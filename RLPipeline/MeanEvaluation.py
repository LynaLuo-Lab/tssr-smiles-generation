"""
Mean evaluation utilities for SMILES generators.

This module aggregates a variety of readability‑oriented helpers and metrics
used to assess batches of generated SMILES. It emphasizes human‑friendly
structure and comments rather than maximum speed.

Highlights
- classify validity: fast checks of syntactic and chemical validity with RDKit
- scaffolds/fragments: compare distributions of Bemis–Murcko scaffolds and
  BRICS fragments between generated and reference sets
- TwoStageReward: same reward logic used by the RL environment for consistency
- MeanEvaluator: orchestrates batch generation, filtering, and reporting of
  mean summary metrics and I/O helpers for writing results to disk
"""
import random
from collections import deque, Counter
import torch
from typing import Any
from rdkit.Chem import SanitizeFlags as sf
import multiprocessing as mp
from RLPipeline.BulkGenerator import Generator
from RLPipeline.SequenceDataSet import LabelEncoder
import itertools
from typing import List, Sequence, Tuple, Set
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cosine
import os

# LRU-cached canonicalization helper to reduce repeated RDKit work
import functools
@functools.lru_cache(maxsize=200000)
def _canon_smiles_or_none(s: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _classify_smi(s: str) -> tuple[bool, bool]:
    """Return (syntactically_valid, chemically_valid) for a SMILES string.
    Optimized: try full sanitization first; if it succeeds, both are True.
    Only fall back to sanitize=False when sanitize=True fails, to detect
    syntactic validity without full chemical sanitization.
    """
    try:
        m2 = Chem.MolFromSmiles(s)  # full sanitize first (fast path)
        if m2 is not None:
            return True, True
        # fall back to syntax-only parse
        m1 = Chem.MolFromSmiles(s, sanitize=False)
        return (m1 is not None), False
    except Exception:
        return False, False

# Indexed variant to preserve ordering when using imap_unordered

def _classify_indexed(args: tuple[int, str]) -> tuple[int, bool, bool]:
    i, s = args
    v, cv = _classify_smi(s)
    return i, v, cv

def bemis_murcko_scaffold(mol: Chem.Mol) -> str:
    """Return the Bemis–Murcko scaffold as a SMILES string (or '')"""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)   # UPDATED call
        return Chem.MolToSmiles(scaffold, isomericSmiles=True) if scaffold else ""
    except Exception:
        return ""

def brics_fragments(mol: Chem.Mol) -> List[str]:
    """Return canonical SMILES of BRICS fragments for a molecule."""
    try:
        frags = Chem.FragmentOnBRICSBonds(mol)
        return sorted(Chem.MolToSmiles(f, True) for f in Chem.GetMolFrags(frags, asMols=True))
    except Exception:
        return []

def _frequency_vector(items: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Given a list of hashable items, return (freq_vector, vocabulary_order)
    where freq_vector is normalised counts (sum==1).
    """
    if not items:
        return np.zeros(0), []
    vocab, counts = np.unique(items, return_counts=True)
    freq_vec = counts / counts.sum()
    return freq_vec.astype(float), vocab.tolist()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray, common_vocab: List[str]) -> float:
    """Cosine similarity on two aligned vectors (handles zero‑vectors)."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    return 1.0 - cosine(vec1, vec2)

def scaffold_similarity(gen_mols: List[Chem.Mol], ref_mols: List[Chem.Mol]) -> float:
    """
    Robust scaffold similarity between two molecule sets based on Bemis–Murcko
    scaffolds using average max Tanimoto similarity over Morgan fingerprints.
    This avoids the near-constant value issue seen with cosine over frequency
    vectors in sparse vocabularies.
    """
    # Extract scaffold molecules for both sets
    gen_scaf_smis = [bemis_murcko_scaffold(m) for m in gen_mols]
    ref_scaf_smis = [bemis_murcko_scaffold(m) for m in ref_mols]
    gen_scaf_mols = [Chem.MolFromSmiles(s) for s in gen_scaf_smis if s]
    ref_scaf_mols = [Chem.MolFromSmiles(s) for s in ref_scaf_smis if s]

    if not gen_scaf_mols or not ref_scaf_mols:
        return 0.0

    # Compute Morgan fingerprints
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in gen_scaf_mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in ref_scaf_mols]

    if not gen_fps or not ref_fps:
        return 0.0

    # Subsample to keep it fast and stable
    max_ref = 1000
    max_gen = 1000
    if len(ref_fps) > max_ref:
        ref_fps = ref_fps[:max_ref]
    if len(gen_fps) > max_gen:
        gen_fps = gen_fps[:max_gen]

    # For each generated scaffold, compute its max similarity to the reference set
    max_sims = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        if sims:
            max_sims.append(max(sims))
    if not max_sims:
        return 0.0
    return float(np.mean(max_sims))

def fragment_similarity(gen_mols: List[Chem.Mol], ref_mols: List[Chem.Mol]) -> float:
    gen_frags = list(itertools.chain.from_iterable(brics_fragments(m) for m in gen_mols))
    ref_frags = list(itertools.chain.from_iterable(brics_fragments(m) for m in ref_mols))
    v1, vocab1 = _frequency_vector(gen_frags)
    v2, vocab2 = _frequency_vector(ref_frags)
    vocab_union = sorted(set(vocab1) | set(vocab2))
    def to_full(vec, vocab_vec):
        full = np.zeros(len(vocab_union))
        if vec.size:
            index_map = {v:i for i,v in enumerate(vocab_vec)}
            for i, term in enumerate(vocab_union):
                if term in index_map:
                    full[i] = vec[index_map[term]]
        return full
    return cosine_similarity(to_full(v1, vocab1), to_full(v2, vocab2), vocab_union)


def build_token_freq() -> np.ndarray:
    """
    Count every token in `path` except the three specials.
    Returns freq[i] aligned with encoder.characters, dtype=float, sum = 1.
    """
    encoder = LabelEncoder()
    counter = Counter()
    specials = {"[BOS]", "[EOS]", "[PAD]"}

    with open('/home/abog/PycharmProjects/Drug-Discovery-Loss-Term/data/train.txt', "r") as f:
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

class TwoStageReward:

    def __init__(self,
                 vocab: list[str],
                 store_threshold: int = 3,
                 cache_size: int | None = None,
                 k_subst: int = 8,):
        self.vocab = vocab
        self.probs = build_token_freq()
        self.k_subst = k_subst
        self.store_thr = store_threshold
        # If cache_size is None or <= 0, use unbounded deques (no cap)
        if cache_size is None or cache_size <= 0:
            self.near_valid_cache = deque()
            self.fixed_cache_before = deque()
            self.fixed_cache_after = deque()
            # Cache pairs (original_unparsable, final_clean) for strings that start unparsable
            # but after Stage 1 (syntax fix) and Stage 2 (chemistry fixes) become fully valid
            # (0 chemistry problems and parse with sanitize=True).
            self.unparsable_to_valid_pairs = deque()
            # Cache pairs (initial_string, fully_fixed_string) for any case that ends with 0 problems
            # (sanitizable), regardless of initial parseability.
            self.initial_to_fully_fixed_pairs = deque()
        else:
            self.near_valid_cache = deque(maxlen=cache_size)
            self.fixed_cache_before = deque(maxlen=cache_size)
            self.fixed_cache_after = deque(maxlen=cache_size)
            # Cache pairs (original_unparsable, final_clean) for strings that start unparsable
            # but after Stage 1 (syntax fix) and Stage 2 (chemistry fixes) become fully valid
            # (0 chemistry problems and parse with sanitize=True).
            self.unparsable_to_valid_pairs = deque(maxlen=cache_size)
            # Cache pairs (initial_string, fully_fixed_string) for any case that ends with 0 problems
            # (sanitizable), regardless of initial parseability.
            self.initial_to_fully_fixed_pairs = deque(maxlen=cache_size)

    def __call__(self, smiles: list) -> tuple[int, int, int]:
        """Compute reward and maybe cache the molecule."""
        initially_working, fixed, mol = self._syntax_and_errors(smiles)
        if initially_working | fixed:
            original_str = ''.join(smiles)
            fail_swaps, problems_change = self._try_reduce_chem_problems(
                mol,
                initially_unparseable=(not initially_working),
                original_str=original_str,
            )
            return int(fixed), fail_swaps, problems_change
        else:
            return int(fixed), 0, 0

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
                self.fixed_cache_before.append(mol)
                self.fixed_cache_after.append(''.join(smil))
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

    def _try_reduce_chem_problems(self, smi: list, *, initially_unparseable: bool = False, original_str: str | None = None) -> tuple[int, int]:
        chars = smi.copy()
        best_mol = Chem.MolFromSmiles(''.join(chars), sanitize=False)
        initial_err = len(Chem.DetectChemistryProblems(best_mol))
        best_err = initial_err

        if 0 < initial_err <= self.store_thr:
            self.near_valid_cache.append(''.join(chars))

        fail_swaps = 0
        # If already no chemistry problems after Stage 1, consider storing immediately
        if best_err == 0:
            final_str = ''.join(chars)
            # If requested, store general initial->fully fixed pair when sanitizable
            if original_str is not None and Chem.MolFromSmiles(final_str) is not None:
                self.initial_to_fully_fixed_pairs.append((original_str, final_str))
            if initially_unparseable and original_str is not None:
                # Check that it sanitizes successfully too
                if Chem.MolFromSmiles(final_str) is not None:
                    self.unparsable_to_valid_pairs.append((original_str, final_str))
            return 0, 0

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
                        final_str = ''.join(chars)
                        # Store general initial->fully fixed pair when sanitizable
                        if original_str is not None and Chem.MolFromSmiles(final_str) is not None:
                            self.initial_to_fully_fixed_pairs.append((original_str, final_str))
                        if initially_unparseable and original_str is not None:
                            if Chem.MolFromSmiles(final_str) is not None:
                                self.unparsable_to_valid_pairs.append((original_str, final_str))
                        return fail_swaps, (initial_err - best_err)
                    break
                else:
                    chars[pos] = original
                    fail_swaps += 1
        return fail_swaps, (initial_err - best_err)



# ─── 4) MEAN EVALUATION: batched + multiprocess ───────────────────────────────

class MeanEvaluator:
    """Compute mean‑level statistics over a batch of generated sequences.

    This class asks a Generator to produce `sample_size` sequences, converts
    them back to SMILES strings using `vocab`, and then computes aggregate
    statistics such as mean length, mean number of chemistry problems, and
    validity/novelty rates. Expensive RDKit work is parallelized when possible.

    Parameters
    - generator: object exposing `generate(n, greedy=False) -> Tensor[n, T]`
    - vocab: list mapping token IDs to their string representation
    - eos_id: token ID used to truncate sequences
    - sample_size: number of sequences to generate for evaluation
    """
    def __init__(
            self,
            generator: Generator,
            vocab: List[str],
            eos_id: int,
            sample_size: int
    ):
        self.gen = generator
        self.vocab = vocab
        self.eos = eos_id
        self.sample = sample_size
        # Generate in manageable chunks to prevent OOM
        sequences = []
        # Configurable generation chunk size to balance memory and throughput
        chunk = int(os.getenv("MEAN_EVAL_GEN_CHUNK", "2048"))
        if self.sample > 0:
            print(f"[MeanEval] Generation chunk size: {chunk}")
        remaining = self.sample
        generated = 0
        if self.sample > 0:
            print(f"[MeanEval] Now generating {self.sample} samples...")
        while remaining > 0:
            bsz = min(chunk, remaining)
            seq_chunk = self.gen.generate(bsz, greedy=False).cpu()
            sequences.append(seq_chunk)
            remaining -= bsz
            generated += bsz
            print(f"[MeanEval] Generated {generated}/{self.sample}")
        if sequences:
            self.seqs = torch.cat(sequences, dim=0).tolist()
        else:
            self.seqs = []
        self.eval_calc = TwoStageReward(vocab, 3, None, 8)
        # Precompute EOS-trimmed token sequences and corresponding SMILES strings for speed
        self.trimmed_token_seqs: List[List[int]] = []
        self.strings: List[str] = []
        for seq in self.seqs:
            if self.eos in seq:
                seq = seq[: seq.index(self.eos)]
            self.trimmed_token_seqs.append(seq)
            self.strings.append(''.join(self.vocab[i] for i in seq))
        # Parallel classification of syntactic and chemical validity to accelerate downstream metrics
        self.valid_flags: List[bool] = []
        self.chem_valid_flags: List[bool] = []
        if self.strings:
            try:
                workers = int(os.getenv("MEAN_EVAL_WORKERS", str(min(8, max(1, (os.cpu_count() or 4) - 1)))))
            except Exception:
                workers = min(8, (os.cpu_count() or 4))
            workers = max(1, workers)
            if workers > 1:
                print(f"[MeanEval] RDKit classify workers: {workers}")
                N = len(self.strings)
                valid_tmp = [False] * N
                chem_valid_tmp = [False] * N
                with mp.get_context("spawn").Pool(processes=workers) as pool:
                    # choose a modest chunksize for better load balancing
                    chksz = max(64, len(self.strings) // (workers * 8) or 1)
                    for i, v, cv in pool.imap_unordered(_classify_indexed, enumerate(self.strings), chunksize=chksz):
                        valid_tmp[i] = v
                        chem_valid_tmp[i] = cv
                self.valid_flags = valid_tmp
                self.chem_valid_flags = chem_valid_tmp
            else:
                for s in self.strings:
                    v, cv = _classify_smi(s)
                    self.valid_flags.append(v)
                    self.chem_valid_flags.append(cv)
            try:
                v_cnt = sum(1 for x in self.valid_flags if x)
                cv_cnt = sum(1 for x in self.chem_valid_flags if x)
                print(f"[MeanEval] Preclassified: valid={v_cnt}, chem_valid={cv_cnt} out of N={len(self.strings)}")
            except Exception:
                pass

    def get_means(self) -> Tuple[float, float, float, float, float]:
        """Compute mean length, problems, swaps, fixed rate, and error change.

        Returns a tuple of 5 floats: (mean_length, mean_problems, mean_swaps,
        mean_fixed, mean_error_change).
        """
        print(f"[MeanEval] Computing mean metrics over {len(self.seqs)} sequences...")
        # compute lengths using pre-trimmed sequences
        lengths = [len(s) for s in self.trimmed_token_seqs]

        # sequential map to (len, problems, swaps) to keep memory safe
        results = []
        for s in self.seqs:
            results.append(self.eval_one(s, self.vocab, self.eos))

        # unzip and average
        _, problems, fixed, swaps, err_change = zip(*results)
        return (
            float(np.mean(lengths)) if lengths else 0.0,
            float(np.mean(problems)) if problems else 0.0,
            float(np.mean(swaps)) if swaps else 0.0,
            float(np.mean(fixed)) if fixed else 0.0,
            float(np.mean(err_change)) if err_change else 0.0,
        )

    def eval_valid_novel(
            self,
            original_smiles: Set[str],
    ) -> Tuple[int, int, int, float, float, float, float]:
        """Compute validity, novelty, and scaffold similarity metrics.

        Returns a 7-tuple:
        (valid_count, novel_count, chem_valid_count, valid_frac, novel_frac,
         chem_valid_frac, scaffold_similarity)

        - valid_count/frac: syntactically valid SMILES (sanitize=False)
        - chem_valid_count/frac: fully sanitizable SMILES (sanitize=True)
        - novel_count/frac: among valid, not present in the provided reference set
        - scaffold_similarity: average max Tanimoto similarity between Bemis–Murcko
          scaffolds of generated vs reference molecules, computed over subsets.
        """
        print(f"[MeanEval] Evaluating validity/novelty on {len(self.seqs)} sequences...")
        # Use precomputed validity flags if present to avoid repeated RDKit parsing
        total = len(self.seqs)
        if getattr(self, "valid_flags", None) and getattr(self, "chem_valid_flags", None) and len(self.valid_flags) == len(self.strings):
            valid_smiles = [s for s, v in zip(self.strings, self.valid_flags) if v]
            chemically_valid_smiles = [s for s, v in zip(self.strings, self.chem_valid_flags) if v]
        else:
            # Fallback: single pass RDKit parsing
            valid_smiles: List[str] = []
            chemically_valid_smiles: List[str] = []
            for smi in self.strings:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol is not None:
                    valid_smiles.append(smi)
                    # chemical validity (full sanitization)
                    mol2 = Chem.MolFromSmiles(smi)
                    if mol2 is not None:
                        chemically_valid_smiles.append(smi)

        valid_count = len(valid_smiles)
        valid_frac = valid_count / total if total else 0.0
        chemically_valid_count = len(chemically_valid_smiles)
        chemically_valid_frac = chemically_valid_count / total if total else 0.0

        # NOVELTY (w.r.t. original set) among syntactically valid
        valid_set = set(valid_smiles)
        novel_set = valid_set - set(original_smiles)
        novel_count = len(novel_set)
        novel_frac = novel_count / valid_count if valid_count > 0 else 0.0

        # --- SIMILARITY METRIC for chemically valid molecules (memory-safe) ---
        # Cap the number of generated and reference SMILES we convert to RDKit mols
        # to avoid large transient memory spikes with very large datasets.
        import gc
        GEN_CAP = int(os.getenv("MEAN_EVAL_SIM_GEN_CAP", "5000"))
        REF_CAP = int(os.getenv("MEAN_EVAL_SIM_REF_CAP", "5000"))
        if len(chemically_valid_smiles) > GEN_CAP:
            print(f"[MeanEval] Similarity prep: capping generated set from {len(chemically_valid_smiles)} to {GEN_CAP}")
            gen_smiles_subset = chemically_valid_smiles[:GEN_CAP]
        else:
            gen_smiles_subset = chemically_valid_smiles

        orig_list = sorted(original_smiles)
        if len(orig_list) > REF_CAP:
            print(f"[MeanEval] Similarity prep: capping reference set from {len(orig_list)} to {REF_CAP}")
            ref_smiles_subset = orig_list[:REF_CAP]
        else:
            ref_smiles_subset = orig_list

        print(f"[MeanEval] Building RDKit mols for similarity: gen={len(gen_smiles_subset)}, ref={len(ref_smiles_subset)}")
        gen_mols: List[Chem.Mol] = []
        for s in gen_smiles_subset:
            m = Chem.MolFromSmiles(s)  # full sanitize OK here; count already small
            if m is not None:
                gen_mols.append(m)
        ref_mols: List[Chem.Mol] = []
        for s in ref_smiles_subset:
            try:
                # sanitize=False is faster and sufficient before scaffold extraction
                m = Chem.MolFromSmiles(s, sanitize=False)
            except Exception:
                m = None
            if m is not None:
                ref_mols.append(m)

        similarity = scaffold_similarity(gen_mols, ref_mols) if gen_mols and ref_mols else 0.0
        # Explicitly free intermediate objects to reduce peak memory and prevent late crashes
        del gen_mols, ref_mols, gen_smiles_subset, ref_smiles_subset
        gc.collect()

        return (valid_count, novel_count, chemically_valid_count, valid_frac, novel_frac, chemically_valid_frac, similarity)
    
    def write_samples_to_file(
            self,
            path_valid,
            path_fixed,
            path_unparsable_to_valid: str | None = None,
            path_initial_to_fully_fixed: str | None = None
    ):
        """
        Persist all cached sample strings to disk without sampling.
        The "amount" parameter is retained for backward compatibility but ignored
        for limiting writes; we always write the full caches if available.
        - path_valid: base path (no extension) for near-valid strings -> writes `${path_valid}.txt`
        - path_fixed: base path for fixed pairs -> writes `${path_fixed}.txt` and `${path_fixed}_fixed.txt`
        - path_unparsable_to_valid: optional TSV of pairs that went from unparsable -> fully valid
        - path_initial_to_fully_fixed: optional TSV of initial -> fully fixed for any case ending valid
        """
        # Write near-valid cache (one string per line)
        nv = list(self.eval_calc.near_valid_cache)
        if nv:
            with open(path_valid + '.txt', 'w') as f:
                for s in nv:
                    f.write(f"{s}\n")
            print(f"[MeanEval] Wrote near-valid strings: {len(nv)} -> {path_valid}.txt")

        # Write fixed before/after pairs fully
        before = list(self.eval_calc.fixed_cache_before)
        after = list(self.eval_calc.fixed_cache_after)
        if before and after:
            with open(path_fixed + '.txt', 'w') as f_before, open(path_fixed + '_fixed.txt', 'w') as f_after:
                for b, a in zip(before, after):
                    f_before.write(f"{b}\n")
                    f_after.write(f"{a}\n")
            print(f"[MeanEval] Wrote fixed pairs: {len(before)} -> {path_fixed}.txt and {path_fixed}_fixed.txt")

        # Write unparsable -> valid pairs (TSV)
        if path_unparsable_to_valid is not None:
            pairs = list(self.eval_calc.unparsable_to_valid_pairs)
            if pairs:
                with open(path_unparsable_to_valid, 'w') as f:
                    for orig, fixed in pairs:
                        f.write(f"{orig}\t{fixed}\n")
                print(f"[MeanEval] Wrote unparsable→valid pairs: {len(pairs)} -> {path_unparsable_to_valid}")

        # Write initial -> fully fixed pairs (TSV)
        if path_initial_to_fully_fixed is not None:
            pairs_all = list(self.eval_calc.initial_to_fully_fixed_pairs)
            if pairs_all:
                with open(path_initial_to_fully_fixed, 'w') as f:
                    for orig, fixed in pairs_all:
                        f.write(f"{orig}\t{fixed}\n")
                print(f"[MeanEval] Wrote initial→fully-fixed pairs: {len(pairs_all)} -> {path_initial_to_fully_fixed}")

    def collect_chemically_valid(self, include_fixed: bool = True) -> list[tuple[str, str]]:
        """
        Return list of (smiles, origin) for all chemically valid molecules from:
        - generated strings (`origin='generated'`)
        - optionally fixed strings from swapping (`origin='fixed'`), including:
          - fixed_cache_after
          - right-hand elements of unparsable_to_valid_pairs and initial_to_fully_fixed_pairs
        Deduplicate by canonical SMILES; prefer generated over fixed on conflicts.
        """
        canon2row: dict[str, tuple[str, str]] = {}
        # Generated (sanitized)
        for s in self.strings:
            can = _canon_smiles_or_none(s)
            if can is None:
                continue
            if can not in canon2row:
                canon2row[can] = (s, 'generated')
        # Fixed additions from all sources
        if include_fixed:
            # 1) fixed_cache_after (ensure chemical validity via canonicalization)
            for s in list(self.eval_calc.fixed_cache_after):
                can = _canon_smiles_or_none(s)
                if can is None:
                    continue
                if can not in canon2row:
                    canon2row[can] = (s, 'fixed')
            # 2) unparsable -> valid pairs (use RHS)
            for _orig, fixed in list(self.eval_calc.unparsable_to_valid_pairs):
                can = _canon_smiles_or_none(fixed)
                if can is None:
                    continue
                if can not in canon2row:
                    canon2row[can] = (fixed, 'fixed')
            # 3) initial -> fully fixed pairs (use RHS)
            for _orig, fixed in list(self.eval_calc.initial_to_fully_fixed_pairs):
                can = _canon_smiles_or_none(fixed)
                if can is None:
                    continue
                if can not in canon2row:
                    canon2row[can] = (fixed, 'fixed')
        return list(canon2row.values())

    def collect_valid_from_pairs(self) -> list[str]:
        """
        Collect the valid (RHS) molecules from both pair caches:
        - unparsable_to_valid_pairs (orig -> fixed)
        - initial_to_fully_fixed_pairs (orig -> fixed)
        Keep only chemically valid (sanitizable) SMILES and deduplicate by canonical SMILES.
        Returns a list of deduplicated SMILES strings (the fixed/original RHS strings are returned).
        """
        from rdkit import Chem
        canon2fixed: dict[str, str] = {}
        # 1) unparsable -> valid pairs (use RHS)
        for _orig, fixed in list(self.eval_calc.unparsable_to_valid_pairs):
            can = _canon_smiles_or_none(fixed)
            if can is None:
                continue
            if can not in canon2fixed:
                canon2fixed[can] = fixed
        # 2) initial -> fully fixed pairs (use RHS)
        for _orig, fixed in list(self.eval_calc.initial_to_fully_fixed_pairs):
            can = _canon_smiles_or_none(fixed)
            if can is None:
                continue
            if can not in canon2fixed:
                canon2fixed[can] = fixed
        return list(canon2fixed.values())

    def write_valid_pairs_list(self, path_txt: str) -> int:
        """
        Write a newline-separated list of all chemically valid molecules collected from
        the RHS of pair caches (unparsable->valid and initial->fully_fixed), deduplicated.
        Returns the number of lines written.
        """
        vals = self.collect_valid_from_pairs()
        if vals:
            with open(path_txt, 'w') as f:
                for s in vals:
                    f.write(f"{s}\n")
            print(f"[MeanEval] Wrote all-valid-from-pairs: {len(vals)} -> {path_txt}")
        else:
            # Ensure an empty file exists for consistency
            with open(path_txt, 'w') as f:
                pass
            print(f"[MeanEval] No valid molecules from pairs; created empty file -> {path_txt}")
        return len(vals)

    def write_all_valid_to_csv(self, path_csv: str, include_fixed: bool = True, enriched: bool = False) -> None:
        """
        Write CSV with chemically valid molecules (deduped by canonical SMILES).
        Columns: smiles, origin; if enriched=True adds canonical_smiles, qed, sa_score, scaffold_smiles.
        """
        import csv
        from rdkit import Chem
        from rdkit.Chem import QED
        from RLPipeline.sa_scorer import calculate_sa_score

        rows = self.collect_chemically_valid(include_fixed=include_fixed)
        # Prepare enrichment if requested
        with open(path_csv, 'w', newline='') as f:
            if enriched:
                writer = csv.writer(f)
                writer.writerow(['smiles', 'origin', 'canonical_smiles', 'qed', 'sa_score', 'scaffold_smiles'])
                for smi, origin in rows:
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        continue
                    can = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
                    try:
                        qed_v = float(QED.qed(m))
                    except Exception:
                        qed_v = float('nan')
                    try:
                        sa_v = float(calculate_sa_score(m))
                    except Exception:
                        sa_v = float('nan')
                    try:
                        from rdkit.Chem.Scaffolds import MurckoScaffold
                        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                    except Exception:
                        scaf = ''
                    writer.writerow([smi, origin, can, qed_v, sa_v, scaf])
            else:
                writer = csv.writer(f)
                writer.writerow(['smiles', 'origin'])
                for smi, origin in rows:
                    writer.writerow([smi, origin])
        print(f"[MeanEval] Wrote valid CSV: {len(rows)} rows -> {path_csv} (enriched={enriched})")

    def compute_additional_metrics(self, valid_smiles: list[str]) -> dict:
        """
        Compute Uniqueness, Diversity (mean NN distance via ECFP4), QED, SA,
        and scaffold diversity stats for the provided chemically valid SMILES.
        Returns a flat dict of metrics.
        """
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, QED
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from statistics import mean, median
        from RLPipeline.sa_scorer import calculate_sa_score
        import random, gc

        mols = []
        can_smis = []
        for s in valid_smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            mols.append(m)
            can_smis.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
        N = len(mols)
        metrics: dict[str, float] = {}
        metrics['valid_count'] = N
        if N == 0:
            # Return zeros/NaNs as appropriate
            metrics.update({
                'uniqueness': 0.0,
                'diversity_mean_nn_distance': 0.0,
                'diversity_sample_size': 0,
                'scaffold_diversity_count': 0,
                'scaffold_diversity_fraction': 0.0,
                'qed_mean': 0.0,
                'qed_median': 0.0,
                'qed_frac_ge_0.6': 0.0,
                'sa_mean': 0.0,
                'sa_median': 0.0,
                'sa_frac_le_3': 0.0,
                'sa_frac_le_5': 0.0,
            })
            return metrics

        # Uniqueness
        uniq = len(set(can_smis))
        metrics['uniqueness'] = uniq / float(N)

        # QED & SA distributions
        qed_vals = []
        sa_vals = []
        for m in mols:
            try:
                qed_vals.append(float(QED.qed(m)))
            except Exception:
                pass
            try:
                sa_vals.append(float(calculate_sa_score(m)))
            except Exception:
                pass
        if qed_vals:
            from statistics import mean as smean, median as smedian
            metrics['qed_mean'] = smean(qed_vals)
            metrics['qed_median'] = smedian(qed_vals)
            metrics['qed_frac_ge_0.6'] = sum(1 for q in qed_vals if q >= 0.6) / len(qed_vals)
        else:
            metrics['qed_mean'] = 0.0
            metrics['qed_median'] = 0.0
            metrics['qed_frac_ge_0.6'] = 0.0
        if sa_vals:
            from statistics import mean as smean, median as smedian
            metrics['sa_mean'] = smean(sa_vals)
            metrics['sa_median'] = smedian(sa_vals)
            metrics['sa_frac_le_3'] = sum(1 for s in sa_vals if s <= 3.0) / len(sa_vals)
            metrics['sa_frac_le_5'] = sum(1 for s in sa_vals if s <= 5.0) / len(sa_vals)
        else:
            metrics['sa_mean'] = 0.0
            metrics['sa_median'] = 0.0
            metrics['sa_frac_le_3'] = 0.0
            metrics['sa_frac_le_5'] = 0.0

        # Scaffold diversity
        scaffolds = []
        for m in mols:
            try:
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
            except Exception:
                scaffolds.append('')
        scaf_set = set([s for s in scaffolds if s])
        metrics['scaffold_diversity_count'] = len(scaf_set)
        metrics['scaffold_diversity_fraction'] = (len(scaf_set) / N) if N else 0.0

        # Diversity via mean nearest-neighbor distance (ECFP4, 2048 bits)
        # Subsample for scalability
        CAP = 2000
        if N > CAP:
            idxs = sorted(random.sample(range(N), CAP))
            mols_cap = [mols[i] for i in idxs]
        else:
            mols_cap = mols
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols_cap]
        M = len(fps)
        if M < 2:
            metrics['diversity_mean_nn_distance'] = 0.0
            metrics['diversity_sample_size'] = M
            return metrics
        nn_dists = []
        for i in range(M):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            if not sims:
                continue
            # ignore self‑similarity (usually 1.0)
            if i < len(sims):
                sims[i] = 0.0
            max_sim = max(sims)
            nn_dists.append(1.0 - max_sim)
        metrics['diversity_mean_nn_distance'] = float(sum(nn_dists) / len(nn_dists)) if nn_dists else 0.0
        metrics['diversity_sample_size'] = M
        # Free memory
        del fps, mols_cap, nn_dists
        gc.collect()
        return metrics

    def write_metrics_csv(self, path_csv: str, metrics: dict) -> None:
        import csv
        with open(path_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'value'])
            for k, v in metrics.items():
                w.writerow([k, v])

    def append_metrics_to_stats(self, stats_path: str, title: str, metrics: dict) -> None:
        with open(stats_path, 'a') as sf:
            sf.write(f"=== {title} ===\n")
            for k, v in metrics.items():
                sf.write(f"{k}: {v}\n")

    def eval_one(
        self,
        seq: List[int],
        vocab: List[str],
        eos_id: int
) -> Tuple[int, int, int, int, int]:
        """
        Return (length, problem_count, fixed, swaps, err_change) for one generated sequence.
        """
        # cut at EOS
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]
        length = len(seq)
        mole = [vocab[i] for i in seq]
        smi = ''.join(mole)
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            pcount = 12
            fixed, swaps, err_change = self.eval_calc(mole)
        else:
            pcount = len(Chem.DetectChemistryProblems(mol, sf.SANITIZE_ALL))
            fixed, err_change, swaps = 0, 0, 0

        return length, pcount, fixed, swaps, err_change



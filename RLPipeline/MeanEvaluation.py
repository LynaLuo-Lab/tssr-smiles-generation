import random
from collections import deque, Counter

import numpy as np
import torch
from rdkit import Chem
from typing import List, Tuple, Set, Any
from rdkit.Chem import SanitizeFlags as sf
import multiprocessing as mp
from RLPipeline.BulkGenerator import Generator
from RLPipeline.SequenceDataSet import LabelEncoder
import itertools, math, warnings, pathlib, functools
from typing import List, Sequence, Tuple, Set
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, rdFMCS
from fcd_torch import FCD
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cosine

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
        ref_fps = random.sample(ref_fps, max_ref)
    if len(gen_fps) > max_gen:
        gen_fps = random.sample(gen_fps, max_gen)

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

class TwoStageReward:

    def __init__(self,
                 vocab: list[str],
                 store_threshold: int = 3,
                 cache_size: int = 1000,
                 k_subst: int = 8,):
        self.vocab = vocab
        self.probs = build_token_freq()
        self.k_subst = k_subst
        self.store_thr = store_threshold
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
        best_err = len(Chem.DetectChemistryProblems(best_mol))
        initial_err = len(Chem.DetectChemistryProblems(best_mol))

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
        chunk = 512
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
        self.eval_calc = TwoStageReward(vocab, 3, 200, 8)
        # Precompute EOS-trimmed token sequences and corresponding SMILES strings for speed
        self.trimmed_token_seqs: List[List[int]] = []
        self.strings: List[str] = []
        for seq in self.seqs:
            if self.eos in seq:
                seq = seq[: seq.index(self.eos)]
            self.trimmed_token_seqs.append(seq)
            self.strings.append(''.join(self.vocab[i] for i in seq))

    def get_means(self) -> Tuple[float, float, float, float, float]:
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
        print(f"[MeanEval] Evaluating validity/novelty on {len(self.seqs)} sequences...")
        # Single pass over precomputed strings to reduce RDKit calls
        valid_smiles: List[str] = []
        chemically_valid_smiles: List[str] = []
        total = len(self.seqs)
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
        GEN_CAP = 5000
        REF_CAP = 5000
        if len(chemically_valid_smiles) > GEN_CAP:
            print(f"[MeanEval] Similarity prep: capping generated set from {len(chemically_valid_smiles)} to {GEN_CAP}")
            gen_smiles_subset = random.sample(chemically_valid_smiles, GEN_CAP)
        else:
            gen_smiles_subset = chemically_valid_smiles

        orig_list = list(original_smiles)
        if len(orig_list) > REF_CAP:
            print(f"[MeanEval] Similarity prep: capping reference set from {len(orig_list)} to {REF_CAP}")
            ref_smiles_subset = random.sample(orig_list, REF_CAP)
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
            amount,
            path_unparsable_to_valid: str | None = None,
            path_initial_to_fully_fixed: str | None = None
    ):  
        if len(self.eval_calc.near_valid_cache) >= amount:
            sampled = random.sample(self.eval_calc.near_valid_cache, amount)
            with open(path_valid+'.txt', "w") as f:
                for s in sampled:
                    f.write(f"{s}\n")
        elif 0 < len(self.eval_calc.near_valid_cache) < amount:
            sampled = random.sample(self.eval_calc.near_valid_cache, len(self.eval_calc.near_valid_cache))
            with open(path_valid+'.txt', "w") as f:
                for s in sampled:
                    f.write(f"{s}\n")
        else:
            pass
        if len(self.eval_calc.fixed_cache_before) >= amount:
            pairs = list(zip(self.eval_calc.fixed_cache_before,
                             self.eval_calc.fixed_cache_after))
            sampled_pairs = random.sample(pairs, amount)
            sampled_before, sampled_after = zip(*sampled_pairs)

            with open(path_fixed + '.txt', "w") as f_before, \
                    open(path_fixed + '_fixed.txt', "w") as f_after:
                for b, a in zip(sampled_before, sampled_after):
                    f_before.write(f"{b}\n")
                    f_after.write(f"{a}\n")

        elif 0 < len(self.eval_calc.fixed_cache_before) < amount:
            pairs = list(zip(self.eval_calc.fixed_cache_before,
                             self.eval_calc.fixed_cache_after))
            sampled_pairs = random.sample(pairs, len(pairs))
            sampled_before, sampled_after = zip(*sampled_pairs)

            with open(path_fixed + '.txt', "w") as f_before, \
                    open(path_fixed + '_fixed.txt', "w") as f_after:
                for b, a in zip(sampled_before, sampled_after):
                    f_before.write(f"{b}\n")
                    f_after.write(f"{a}\n")
        else:
            pass
        # Write the new collection if requested
        if path_unparsable_to_valid is not None:
            pairs = list(self.eval_calc.unparsable_to_valid_pairs)
            if len(pairs) > 0:
                if len(pairs) > amount:
                    pairs = random.sample(pairs, amount)
                with open(path_unparsable_to_valid, "w") as f:
                    for orig, fixed in pairs:
                        f.write(f"{orig}\t{fixed}\n")
        # Write general initial -> fully fixed pairs if requested
        if path_initial_to_fully_fixed is not None:
            pairs_all = list(self.eval_calc.initial_to_fully_fixed_pairs)
            if len(pairs_all) > 0:
                if len(pairs_all) > amount:
                    pairs_all = random.sample(pairs_all, amount)
                with open(path_initial_to_fully_fixed, "w") as f:
                    for orig, fixed in pairs_all:
                        f.write(f"{orig}\t{fixed}\n")

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



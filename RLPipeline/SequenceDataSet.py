"""
Efficient text dataset and tokenizer utilities for SMILES/SELFIES sequences.

This module defines two main components:
- LabelEncoder: a regex‑based tokenizer for a fixed SMILES vocabulary with
  robust encode/decode helpers.
- SequenceDataset: a memory‑efficient line dataset that uses a precomputed
  byte offset index (.idx sidecar file) to seek lines on demand. It yields
  (input, target) pairs suitable for next‑token prediction.

Design goals:
- Multiprocessing‑safe lazy file access in __getitem__ to support DataLoader workers.
- Minimal memory footprint even for large files, thanks to the offsets index.
- Optional SELFIES mode that builds its vocabulary from the data.
"""
import os
import re
from typing import List, Tuple
import selfies as sf
import torch
from torch.utils.data import Dataset


class LabelEncoder:
    """Simple encoder that maps each symbol in `characters` to an integer label.

    Notes
    - The `characters` list contains multi‑character SMILES tokens (e.g. "Cl",
      "Br") and single characters. Tokenization is done via a regex that
      prioritizes longer tokens first to avoid splitting multi‑char symbols.
    - Special tokens are [BOS] (begin‑of‑sequence), [EOS] (end‑of‑sequence),
      and [PAD] (padding). They are part of the vocabulary but typically not
      present in raw training text.
    """

    # Ordered, non‑overlapping vocabulary.  If you need to add symbols, just
    # extend this list; the indices are derived automatically.
    characters: List[str] = [
        "[BOS]", "Br", "N", ")", "c", "o", "6",
        "s", "Cl", "=", "2", "]", "C", "n", "O",
        "4", "1", "#", "S", "F", "3", "[", "5",
        "H", "(", "-", "[EOS]", "[PAD]",
    ]
    _special = {"[PAD]", "[BOS]", "[EOS]"}

    def __init__(self):
        # character → integer and inverse
        self.cti = {ch: i for i, ch in enumerate(self.characters)}
        self.itc = {i: ch for ch, i in self.cti.items()}
        # Regex tokenizer compiled lazily
        self._token_re = None

    @property
    def vocab_size(self) -> int:
        """Number of symbols in the vocabulary."""
        return len(self.characters)

    def encode(self, text: str) -> torch.Tensor:
        """Return a 1‑D LongTensor with integer labels for `text`.

        The function tokenizes the input with a regex that matches the longest
        valid tokens first, which is important for multi‑character tokens like
        "Cl" and "Br".
        """
        tokens = self.tokenize(text.strip())
        return torch.tensor([self.cti[t] for t in tokens], dtype=torch.long)

    def decode(self, labels: torch.Tensor) -> str:
        """Inverse of :py:meth:`encode`.  Works on 1‑D tensors."""
        return "".join(self.itc[int(idx)] for idx in labels)

    @property
    def special_ids(self) -> set[int]:
        """{int indices} of PAD/BOS/EOS inside the vocabulary."""
        return {self.cti[tok] for tok in self._special}

    def is_special(self, token: str) -> bool:
        """Return True if `token` is one of the special markers."""
        return token in self._special

    def tokenize(self, text: str) -> List[str]:
        """Split `text` into SMILES tokens using a compiled regex."""
        if self._token_re is None:
            pattern = "|".join(sorted(map(re.escape, self.characters), key=len, reverse=True))
            self._token_re = re.compile(pattern)
        return self._token_re.findall(text)


class SequenceDataset(Dataset):
    """Line‑by‑line text dataset that returns *(input, target)* label tensors.

    Example
    >>> ds = SequenceDataset("data/smiles.txt")
    >>> x, y = ds[0]  # x, y are 1‑D LongTensors of equal length

    In SELFIES mode, the vocabulary is built from the dataset and the
    encoding uses `selfies_to_encoding`.
    """

    def __init__(self, path: str, *, is_selfies: bool = False):
        self.path = path
        self.is_selfies = is_selfies

        # Build or load a lightweight byte‑offset index for fast random access
        idx_path = f"{path}.idx"
        if os.path.exists(idx_path):
            # Load index tensor safely; weights_only=True avoids pickle usage in new PyTorch versions
            self.offsets = torch.load(idx_path, weights_only=True)
        else:
            offsets, offset = [], 0
            with open(path, "rb") as fh:
                for line in fh:
                    offsets.append(offset)
                    offset += len(line)
            self.offsets = torch.tensor(offsets, dtype=torch.long)
            torch.save(self.offsets, idx_path)

        # Read the raw lines to optionally inspect SELFIES length/vocabulary
        with open(path, "r", encoding="utf‑8") as fh:
            raw_lines = [ln.strip() for ln in fh]

        if is_selfies:
            # Dynamically build a vocabulary from the SELFIES alphabet
            alphabet = sf.get_alphabet_from_selfies(raw_lines)
            alphabet.add("[PAD]")
            self.alphabet = sorted(alphabet)
            self.sym2idx = {s: i for i, s in enumerate(self.alphabet)}
            # Longest SELFIES length for optional padding/use (not applied here)
            self.pad_to_len = max(sf.len_selfies(s) for s in raw_lines) if raw_lines else 0
            self.vocab_size = len(self.alphabet)
        else:
            self.encoder = LabelEncoder()
            self.vocab_size = self.encoder.vocab_size

        # file handle will be opened lazily inside __getitem__ (important for workers)
        self._fh = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy open (multiprocessing‑safe)
        if self._fh is None:
            self._fh = open(self.path, "r", encoding="utf‑8")

        # Seek + read one logical line
        self._fh.seek(int(self.offsets[idx]))
        line = self._fh.readline().strip()

        # Encode to label tensor
        labels = self._encode(line)

        # Predict next‑token distribution at each position
        return labels[:-1], labels[1:]

    def _encode(self, text: str) -> torch.Tensor:
        """Encode a single string to integer labels depending on mode."""
        if self.is_selfies:
            labels, _ = sf.selfies_to_encoding(
                text,
                vocab_stoi=self.sym2idx,
                pad_to_len=-1,
                enc_type="labels",
            )
            return torch.tensor(labels, dtype=torch.long)
        else:
            return self.encoder.encode(text)

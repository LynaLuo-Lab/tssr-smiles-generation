import os
import re
from typing import List, Tuple
import selfies as sf
import torch
from torch.utils.data import Dataset


class LabelEncoder:
    """Simple encoder that maps each symbol in `characters` to an integer label.
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

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> torch.Tensor:
        """Return a 1‑D LongTensor with integer labels for `text`."""
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
        return token in self._special
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        if self._token_re is None:
            pattern = "|".join(sorted(map(re.escape, self.characters), key=len, reverse=True))
            self._token_re = re.compile(pattern)
        return self._token_re.findall(text)


class SequenceDataset(Dataset):
    """Line‑by‑line text dataset that returns *(input, target)* label tensors.

    ```python
    ds = SequenceDataset("data/smiles.txt")
    x, y = ds[0]  # x, y are 1‑D LongTensors of equal length
    ```
    """

    def __init__(self, path: str, *, is_selfies: bool = False):
        self.path = path
        self.is_selfies = is_selfies

        # ------------------------------------------------------------------
        # Build file‑offset index so we can perform O(1) random access per line
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Prepare encoders
        # ------------------------------------------------------------------
        with open(path, "r", encoding="utf‑8") as fh:
            raw_lines = [ln.strip() for ln in fh]

        if is_selfies:
            # Dynamically build a vocabulary from the SELFIES alphabet
            alphabet = sf.get_alphabet_from_selfies(raw_lines)
            alphabet.add("[PAD]")
            self.alphabet = sorted(alphabet)
            self.sym2idx = {s: i for i, s in enumerate(self.alphabet)}
            self.pad_to_len = max(sf.len_selfies(s) for s in raw_lines)
            self.vocab_size = len(self.alphabet)
        else:
            self.encoder = LabelEncoder()
            self.vocab_size = self.encoder.vocab_size

        # file handle will be opened lazily inside __getitem__
        self._fh = None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy open (multiprocessing‑safe)
        if self._fh is None:
            self._fh = open(self.path, "r", encoding="utf-8")

        # Seek + read one logical line
        self._fh.seek(int(self.offsets[idx]))
        line = self._fh.readline().strip()

        # Encode to label tensor
        labels = self._encode(line)

        # Predict next‑token distribution at each position
        return labels[:-1], labels[1:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
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

"""
Utilities for sequence generation and basic validity checks using RDKit.

This module contains:
- sanitize: robustly checks if a SMILES string can be parsed/sanitized by RDKit.
- Validator: reads a generated file, filters valid & unique SMILES, and reports stats.
- Generator: a sampling helper for character‑level RNN models with optional top‑p sampling
  and n‑gram based warm‑starts.

Notes for readers:
- SMILES validity here means RDKit can build a molecule object (MolFromSmiles).
- Chemical validity uses RDKit sanitization; syntax‑only parsing uses sanitize=False.
"""
import torch, torch.nn as nn, numpy as np, pandas as pd
from rdkit import RDLogger, Chem

def sanitize(smiles: str) -> bool:
    """Return True if RDKit can parse and sanitize the SMILES string.

    This trims whitespace, then attempts a full RDKit sanitization which checks
    for both syntactic and basic chemical validity.
    """
    smi = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return mol is not None
    except Exception as e:
        print(f"Error sanitizing SMILES '{smi}': {e}")
        return False


class Validator:
    """Validate generated SMILES files and compute simple uniqueness/novelty stats.

    Parameters
    - file_path: path to a text file with one generated SMILES per line
    - valid_path: path to write back only the valid and unique SMILES
    """
    def __init__(self, file_path: str, valid_path: str):
        self.valid_path = valid_path
        self.file_path = file_path
        RDLogger.DisableLog('rdApp.*')

    def validate_generation(self) -> None:
        """Read generated lines, keep only valid/unique SMILES, and write them out."""
        valid_count = 0
        total = 0
        valid = []
        with open(self.file_path, 'r') as f:
            for line in f:
                total += 1
                if sanitize(line) :
                    valid_count += 1
                    if line.strip() not in valid and line.strip() != '':
                        valid.append(line.strip())
        print(f"Valid %: {valid_count/total * 100}")
        with open(self.valid_path, 'w') as file:
            for item in valid:
                file.write(f"{item.strip()}\n")

    def generate_stats(self):
        """Compute and print simple uniqueness and novelty percentages.

        - Uniqueness: fraction of unique valid SMILES relative to all generated.
        - Novelty: fraction of valid SMILES not present in the original training set.
        Reads 3 files: training set (../data/train.csv), the filtered valid file
        `self.valid_path`, and the raw generated file `self.file_path`.
        """
        with open('../data/train.csv', 'r') as f:
            original_lines = f.read().splitlines()
        with open(self.valid_path, 'r') as f:
            valid_lines = f.read().splitlines()
        with open(self.file_path, 'r') as f:
            generated_lines = f.read().splitlines()
        actual_unique = len(valid_lines)/len(generated_lines) * 100
        novel = 0
        for line in valid_lines:
            if line not in original_lines:
                novel += 1
        novel_rate = novel/len(valid_lines) * 100
        print(f"Percent of Unique lines: {actual_unique:.2f}%")
        print(f'Percentage of Unique lines that are novel: {novel_rate:.2f}%')
class Generator:
    """Character‑RNN based sampler with optional n‑gram warm start and top‑p.

    Parameters
    - char_rnn: trained language model; must implement .eval(), .to(), .init_hidden(), and __call__ returning (logits, state)
    - endecode: object exposing encode/decode helpers between tokens and vectors
    - vocab_size: size of the model vocabulary (for building one‑hot vectors)
    - n_gram: if >1, seeds generation with a common (n‑1)‑gram from training data
    - p: nucleus sampling parameter for top‑p filtering
    - temp: temperature (not used directly in the current implementation)
    """
    def __init__(self, char_rnn, endecode, vocab_size, n_gram, p, temp):
        self.charRNN = char_rnn.eval()
        self.endecode = endecode
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.p = p
        self.temp = temp
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def top_p_filtering(self, logits_p):
        """Apply nucleus (top‑p) sampling to the last‑time‑step logits.

        Returns the index of the next token sampled from the filtered
        probability distribution.
        """
        probs = nn.functional.softmax(logits_p.squeeze(0)[-1] / self.p, dim=0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        filtered_probs = probs.masked_fill(indices_to_remove, 0).clone()
        filtered_probs = filtered_probs / filtered_probs.sum()
        next_token_idx = torch.multinomial(filtered_probs, 1).item()
        return next_token_idx

    def get_compound_token(self,s):
        """Extract an (n-1)-length compound token honoring multi-char symbols.

        This scans the string left-to-right and builds a token of length
        `self.n_gram - 1`, preferring 'Cl' and 'Br' over single characters.
        Returns an empty string for edge cases (non-string, empty, or n<=1).
        """
        n = self.n_gram - 1
        if not isinstance(s, str) or not s or n <= 0:
            return ""

        token_parts = []
        current_length = 0
        string_index = 0

        while current_length < n and string_index < len(s):
            if s[string_index:].startswith('Cl'):
                token_parts.append('Cl')
                current_length += 1
                string_index += 2
            elif s[string_index:].startswith('Br'):
                token_parts.append('Br')
                current_length += 1
                string_index += 2
            else:
                token_parts.append(s[string_index])
                current_length += 1
                string_index += 1

        return "".join(token_parts)

    def generate(self, filepath, amount):
        """Generate `amount` SMILES and write one per line to `filepath`.

        Uses top‑p sampling at each step. If n_gram > 1, the first (n‑1) tokens
        are seeded from a high‑frequency compound token mined from the training
        data; otherwise starts from [BOS] only. Stops each sequence at [EOS] or
        when a max character cap is reached to avoid runaway samples.
        """
        if self.n_gram == 1:
            current_n_gram = self.endecode.encode('[BOS]').to(self.device)
        else:
            string_series = pd.read_csv('../data/train.csv', header=None)[0]
            string_series = string_series[string_series.apply(lambda x: isinstance(x, str) and x != '')]
            top_n_grams = string_series.apply(lambda s: self.get_compound_token(s))
            top_chars = (top_n_grams.value_counts() / sum(top_n_grams.value_counts())).to_dict()
            token = np.random.choice(list(top_chars.keys()), p=list(top_chars.values()))
            start_token = self.endecode.encode('[BOS]')
            current_n_gram = self.endecode.encode_sequence(token, skip_append=True)
            current_n_gram = torch.tensor(np.concatenate((start_token, current_n_gram), axis=0)).to(self.device)

        self.charRNN.to(self.device)
        self.charRNN.eval()
        generations = []
        i = 0
        while i < amount:
            generation = []
            charCount = 0
            print(f"Generation {i + 1}/{int(amount)}", end='\r')
            with torch.no_grad():
                hidden = self.charRNN.init_hidden(1, self.device)
                while True:
                    while current_n_gram.dim() < 4:
                        current_n_gram = current_n_gram.unsqueeze(0)
                    current_n_gram = current_n_gram.to(self.device)
                    logits, hidden = self.charRNN(current_n_gram, hidden)
                    next_idx = self.top_p_filtering(logits)
                    next_vec = torch.zeros(self.vocab_size, device=current_n_gram.device)
                    next_vec[next_idx] = 1
                    next_vec = next_vec.view(1, 1, -1)
                    char = self.endecode.decode(next_vec.squeeze(0))
                    charCount += 1
                    if char == '[EOS]' or charCount >= 400:
                        break
                    generation.append(char)
                    current_n_gram = next_vec.to(self.device)
            generations.append(''.join(generation))
            if generations[-1] == '' or generations[-1] == '\n' or generations[-1] == ' ':
                generations.pop()
                i -= 1
            i += 1
        with open(filepath, 'w') as file:
            for item in generations:
                    file.write(f"{item}\n")
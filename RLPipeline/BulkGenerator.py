import torch, torch.nn as nn
import torch.nn.functional as F
from typing import List

class Generator:
    def __init__(self, model: nn.Module, bos_id: int, eos_id: int, max_len: int):
        self.model = model
        self.bos = bos_id
        self.eos = eos_id
        self.max_len = max_len

    def generate(self,n, greedy: bool = False) -> torch.Tensor:
        """
        Returns a [n, max_len] tensor of token‐IDs.
        If greedy=True uses argmax; otherwise samples from the softmax.
        Uses torch.no_grad() to avoid building autograd graphs during sampling.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            # 1) init hidden state
            h = self.model.init_hidden(batch_size=n, device=device)

            # 2) start all sequences with <BOS>
            tokens = torch.full((n, 1), self.bos, dtype=torch.long, device=device)

            # 3) buffer to hold all outputs
            sequences = torch.zeros((n, self.max_len), dtype=torch.long, device=device)

            for t in range(self.max_len):
                # forward one step
                logits, h = self.model(tokens, h)   # logits: [n, 1, V]
                logits = logits[:, -1, :]           # [n, V]

                if greedy:
                    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [n,1]
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)     # [n,1]

                sequences[:, t] = next_tokens.squeeze(1)
                tokens = next_tokens

                # optional: stop early if *all* hit EOS
                if (next_tokens == self.eos).all():
                    break

        return sequences

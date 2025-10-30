"""
Bulk sequence generator for autoregressive token models.

This module provides a very small, dependency‑free wrapper around a PyTorch
sequence model to generate batches of token IDs either greedily (argmax) or via
sampling from the softmax distribution. It intentionally does not perform any
post‑processing; the caller is expected to interpret EOS/PAD tokens.

Key points for readers:
- Input: a model with signature logits, state = model(tokens, state)
  where `tokens` is [batch, time] of integer token IDs and logits are
  [batch, time, vocab_size].
- Output: a tensor of shape [n, max_len] containing generated token IDs.
- The first step is seeded with a single [BOS] token for every sequence.
- Generation runs under torch.inference_mode() to avoid autograd overhead.
"""
import torch, torch.nn as nn
import torch.nn.functional as F
class Generator:
    """Lightweight batch generator for autoregressive models.

    Parameters
    - model: nn.Module implementing forward(tokens, state) -> (logits, new_state)
    - bos_id: integer ID of the [BOS] token to start each sequence
    - eos_id: integer ID of the [EOS] token used to optionally stop early
    - max_len: maximum number of tokens to sample per sequence
    """
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
        with torch.inference_mode():
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

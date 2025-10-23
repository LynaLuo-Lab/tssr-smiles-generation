import copy
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def confidence_penalty(logits, beta=0.1):
    P = F.softmax(logits, dim=-1)
    H = -(P * torch.log(P + 1e-12)).sum(dim=-1)
    return -beta * H.mean()

class CharRNNModel(LightningModule):
    def __init__(self, vocab_size, num_layers, pad_idx, lr=None, hidden_size=1024, dropout=0.2, embedding_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.GRU = nn.GRU(
            self.hparams.embedding_dim,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        self.linear = nn.Linear(self.hparams.hidden_size * 2, self.hparams.vocab_size)
        self._init_gru_weights()

    def forward(self, x, state=None, info = {}):
        hidden = state
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        device = next(self.parameters()).device
        if x.dtype in (torch.float, torch.double):
            x = x.argmax(dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = x.long().to(device)
        x = x.flatten(start_dim=1)
        if hidden is None:
            hidden = self.init_hidden(x.size(0),device=device)
        else:
            if hidden.dim() == 3 and hidden.size(0) != self.hparams.num_layers:
                hidden = hidden.permute(1, 0, 2).contiguous()
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        x, hidden = self.GRU(x_embed, hidden)
        cat = torch.cat([x, hidden[-1].unsqueeze(1).expand(-1, x.size(1), -1), ], dim=-1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        return self.linear(cat), hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = y
        hidden = self.init_hidden(x.size(0), x.device)
        logits, _ = self(x, hidden)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target, reduction='mean', ignore_index=self.hparams.pad_idx)
        conf_pen = confidence_penalty(logits, beta=0.1)
        loss += conf_pen
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        target = y
        hidden = self.init_hidden(x.size(0), x.device)
        logits, _ = self(x, hidden)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target, reduction='mean', ignore_index=self.hparams.pad_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max = 10
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.hparams.num_layers, batch_size, self.hparams.hidden_size,device=device)

    def _init_gru_weights(self):
        for name, param in self.GRU.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

class Critic(nn.Module):
    def __init__(self, model):
        super().__init__()
        H = model.hparams.hidden_size
        V = model.hparams.vocab_size
        self.model = copy.deepcopy(model)
        self.v_head = nn.Sequential(
            nn.Linear(H + V, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, H // 4),
            nn.ReLU(),
            nn.Linear(H // 4, 1),
        )

    def forward(self, x):
        x, hidden = self.model(x)
        x = torch.cat([x, hidden[:, -1, :].unsqueeze(1).expand(-1, x.size(1), -1), ], dim=-1)
        x = self.v_head(x)
        return x.squeeze(-1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCHead(nn.Module):
    def __init__(self, encoder_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, vocab_size+1)

    def forward(self, encoder_out):
        """
        encoder_out: (B, T_enc, D)
        returns log_probs: (T_enc, B, V) for CTC
        """
        logits = self.proj(encoder_out)              # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)              # (T, B, V)
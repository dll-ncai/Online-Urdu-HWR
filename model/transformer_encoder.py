import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class TrainablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        T = x.size(1)
        return x + self.pos[:, :T, :]

class RobertaEncoder(nn.Module):
    """A Transformer encoder based on RobertaModel."""
    def __init__(self, d_model, nhead, num_layers, intermediate_size, dropout=0.1, vocab_size=50265):
        super().__init__()
        config = RobertaConfig(
            vocab_size=vocab_size, # or a larger default from roberta
            hidden_size=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.transformer_encoder = RobertaModel(config)
        # We only use the encoder part of RobertaModel
        self.d_model = d_model
        self.pos_embedding = TrainablePositionalEmbedding(
            d_model, max_len=512
        )

    def forward(self, src, attention_mask=None):
        """
        Forward pass of the Transformer encoder.
        Args:
            src (torch.Tensor): Input tensor from CNN of shape (B, T, D).
            src_key_padding_mask (torch.Tensor): Padding mask for src.
        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        
        # RobertaModel takes inputs_embeds
        src = self.pos_embedding(src)
        output = self.transformer_encoder(inputs_embeds=src, attention_mask=attention_mask)
        return output.last_hidden_state

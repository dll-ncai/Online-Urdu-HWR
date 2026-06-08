import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import math


def sinusoidal_positions(n_positions, dim):
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe




def get_decoder(vocab_size, d_model, nhead, num_layers, n_positions, bos_token_id, eos_token_id, decoder_path=None, freeze=False):
    """
    Loads a pretrained GPT-2 style decoder.
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=d_model,
        n_head=nhead,
        n_layer=num_layers,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        add_cross_attention=True,
    )
    decoder = GPT2LMHeadModel(config)
    if decoder_path:
        decoder.load_state_dict(
            GPT2LMHeadModel.from_pretrained(decoder_path).state_dict(),strict=False
        )
    
    # Replace decoder positional embedding
    decoder.transformer.wpe = nn.Embedding.from_pretrained(
        sinusoidal_positions(config.n_positions, config.n_embd),
        freeze=True
    )
    
    if freeze:
        for param in decoder.parameters():
            param.requires_grad = False
            
    return decoder

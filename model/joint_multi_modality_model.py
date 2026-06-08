import torch
import torch.nn as nn
from .mulit_modality import MultiModalCNNEncoder3
from .transformer_encoder import RobertaEncoder
from .transformer_decoder import get_decoder
from .ctc_head import CTCHead

class JointMultiModel(nn.Module):
    """The combined model with CNN, Transformer encoder, decoder, and CTC head."""
    def __init__(self, trans_enc_d_model, trans_enc_nhead, trans_enc_layers, trans_enc_ff_dim,
                 tokenizer, trans_dec_d_model, trans_dec_nhead, trans_dec_layers, trans_dec_n_positions, freeze_decoder=True, encoder_path=None, decoder_path=None, cnn_encoder_path=None, ctc_head_path=None, img_feat=None, aux_feat=None, freeze_pcnn_encoder=False, freeze_tr_encoder=False, fusion_type="adaptive"):
        super().__init__()
        self.cnn_encoder = MultiModalCNNEncoder3(cnn_encoder_path=cnn_encoder_path, img_feat=img_feat, aux_feat=aux_feat, freeze_pcnn_encoder=freeze_pcnn_encoder, fusion_type=fusion_type)
        self.transformer_encoder = RobertaEncoder(d_model=trans_enc_d_model, nhead=trans_enc_nhead, 
                                                      num_layers=trans_enc_layers, intermediate_size=trans_enc_ff_dim)
        self.transformer_encoder.load_state_dict(torch.load(encoder_path))
        print(f"Loaded Transformer encoder weights from {encoder_path}")
        self.transformer_decoder = get_decoder(vocab_size=tokenizer.vocab_size, d_model=trans_dec_d_model, 
                                               nhead=trans_dec_nhead, num_layers=trans_dec_layers,n_positions=trans_dec_n_positions, bos_token_id=tokenizer.bos_token_id, eos_token_id=None,
                                               decoder_path=None, freeze=freeze_decoder)
        self.transformer_decoder.load_state_dict(torch.load(decoder_path))
        print(f"Loaded Transformer decoder weights from {decoder_path}")
        self.ctc_head = CTCHead(encoder_dim=trans_enc_d_model, vocab_size=tokenizer.vocab_size)
        self.ctc_head.load_state_dict(torch.load(ctc_head_path))
        print(f"Loaded CTC head weights from {ctc_head_path}")
        
        if freeze_tr_encoder:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            print("Transformer encoder frozen.")
        
        # Projection layer if CNN output and Transformer input dimensions don't match
        if 256 != trans_enc_d_model:
            self.projection = nn.Linear(256, trans_enc_d_model)
        else:
            self.projection = nn.Identity()

    def forward(self, pixel_values, labels, attention_mask):
        """
        pixel_values: dict of tensors - input images for different modalities
        labels: (B, T_dec) - target token ids for decoder
        attention_mask: (B, T_dec) - mask for decoder input
        """
        # cnn_output = self.cnn_encoder(pixel_values, ablation_mode="img_only")
        cnn_output = self.cnn_encoder(pixel_values)
        # print("CNN Output shape:", cnn_output.shape)
        
        projected_output = self.projection(cnn_output)
        # print("Projected Output shape:", projected_output.shape)
        
        encoder_output = self.transformer_encoder(projected_output)
        # print("Encoder Output shape:", encoder_output.shape)
        
        # For CTC head
        ctc_log_probs = self.ctc_head(encoder_output)
        # print("CTC Log Probs shape:", ctc_log_probs.shape)
        
        decoder_output = self.transformer_decoder(
            input_ids=labels,
            encoder_hidden_states=encoder_output,
            attention_mask=attention_mask,
        )
        # print("Decoder Output shape:", decoder_output.logits.shape)
        
        return ctc_log_probs, decoder_output.logits

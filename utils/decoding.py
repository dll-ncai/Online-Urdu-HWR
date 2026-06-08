import torch
from transformers import GenerationConfig

def greedy_decode(model, pixel_values, max_len, tokenizer, device):
    """
    Greedy decoding for inference.
    """
    model.eval()
    size = 0
    if isinstance(pixel_values, dict):
        for key in pixel_values:
            pixel_values[key] = pixel_values[key].to(device)
            size = pixel_values[key].size(0)
    elif isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device)
        size = pixel_values.size(0)
    with torch.no_grad():
        cnn_output = model.cnn_encoder(pixel_values)
        projected_output = model.projection(cnn_output)
        encoder_output = model.transformer_encoder(projected_output)

        # Start with BOS token
        tgt = torch.full((size, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)

        for _ in range(max_len):
            decoder_output = model.transformer_decoder(input_ids=tgt, encoder_hidden_states=encoder_output)
            next_token_logits = decoder_output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)
    return tgt

def beam_search_decode(model, pixel_values, tokenizer, device="cpu"):
    model.eval()
    if isinstance(pixel_values, dict):
        for key in pixel_values:
            pixel_values[key] = pixel_values[key].to(device)
    elif isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device)

    with torch.no_grad():

        # ---- Encode image ----
        cnn_out = model.cnn_encoder(pixel_values)
        proj_out = model.projection(cnn_out)
        encoder_outputs = model.transformer_encoder(proj_out)

        # ---- HuggingFace generation config ----
        gen_config = GenerationConfig(
            max_length=256,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True,
            no_repeat_ngram_size=0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        outputs = model.transformer_decoder.generate(
            inputs=None,
            encoder_hidden_states=encoder_outputs,
            generation_config=gen_config,
        )

    return outputs
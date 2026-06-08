from transformers import GPT2Tokenizer


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("vocabs/ved/")
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<pad>'
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'
    return tokenizer
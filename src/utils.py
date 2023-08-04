import torch


def decode_from_logits(tokenizer, logits: torch.Tensor):
    return tokenizer.batch_decode(torch.argmax(logits, dim=2).to("cpu"))

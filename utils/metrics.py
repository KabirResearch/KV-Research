import torch
import torch.nn.functional as F

def compute_loss(logits, labels, tokenizer):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )

def compute_entropy(logits):
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()

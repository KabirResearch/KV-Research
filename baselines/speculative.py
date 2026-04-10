"""
Speculative Decoding Baseline
================================
A small draft model generates K candidate tokens quickly.
The larger target model scores all K tokens in a single forward pass.
Tokens are accepted/rejected via a rejection sampling scheme.
Accepted tokens are appended; on rejection, a corrected token is resampled.

Pseudocode:
    while not done:
        draft_tokens = draft_model.generate(K)
        target_logits = target_model.score(context + draft_tokens)
        accepted = []
        for k in range(K):
            ratio = target_prob[k] / draft_prob[k]
            if uniform() < min(1, ratio):
                accepted.append(draft_tokens[k])
            else:
                accepted.append(sample from corrected target dist)
                break
        tokens.extend(accepted)

Paper: Speculative Decoding — Leviathan et al. 2023
       https://arxiv.org/abs/2211.17192
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    K: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
):
    """
    Generate text using speculative decoding.

    Args:
        draft_model: small fast model for drafting
        target_model: large accurate model for verification
        tokenizer: shared tokenizer
        prompt: input text
        K: number of draft tokens per iteration
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
    Returns:
        generated text string
    """
    device = next(target_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens // K + 1):
        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break

        # --- Step 1: Draft K tokens ---
        draft_input = generated
        draft_ids = []
        draft_probs_list = []
        for _ in range(K):
            draft_out = draft_model(draft_input)
            logits = draft_out.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            draft_ids.append(token)
            draft_probs_list.append(probs.gather(1, token))
            draft_input = torch.cat([draft_input, token], dim=-1)

        draft_tokens = torch.cat(draft_ids, dim=-1)  # [1, K]

        # --- Step 2: Target scores context + all K draft tokens in one pass ---
        full_input = torch.cat([generated, draft_tokens], dim=-1)
        target_out = target_model(full_input)
        target_logits = target_out.logits[:, generated.shape[1] - 1: -1, :] / temperature
        target_probs = F.softmax(target_logits, dim=-1)  # [1, K, vocab]

        # --- Step 3: Acceptance sampling ---
        accepted = []
        for k in range(K):
            draft_p = draft_probs_list[k]  # [1, 1]
            target_p = target_probs[:, k, :].gather(1, draft_ids[k])  # [1, 1]
            ratio = (target_p / (draft_p + 1e-9)).clamp(max=1.0)
            if torch.rand(1, device=device).item() < ratio.item():
                accepted.append(draft_ids[k])
            else:
                # Resample from corrected distribution
                corrected = (target_probs[:, k, :] - F.softmax(
                    draft_model(torch.cat([generated, *accepted], dim=-1) if accepted
                               else generated).logits[:, -1, :] / temperature, dim=-1
                )).clamp(min=0)
                corrected = corrected / (corrected.sum() + 1e-9)
                resampled = torch.multinomial(corrected, 1)
                accepted.append(resampled)
                break

        if accepted:
            generated = torch.cat([generated] + accepted, dim=-1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

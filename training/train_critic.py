"""
Attention-Supervised Critic Training
=======================================
Trains the LogTemporalCritic using hidden states and attention weights
from a frozen base model. The target signal is the mean attention
received by each token across TARGET_LAYERS — a proxy for token importance.

Pseudocode:
    for each batch:
        h, attentions = frozen_model(input_ids)
        h_block = mean(h[TARGET_LAYERS])            # block-level hidden
        target = mean_attn_sum(attentions[TARGET_LAYERS])  # attention supervision
        pred = critic(h_block)
        loss = MSE(pred, target)
        loss.backward(); optimizer.step()
"""

import torch
import torch.nn.functional as F
import logging

try:
    import wandb

    _WANDB = True
except ImportError:
    _WANDB = False
from models.critics import LogTemporalCritic
from logs.research_logger import log_event

logger = logging.getLogger(__name__)

TARGET_LAYERS = [10, 12, 14]  # Adjust for model depth


def train_block_critic(
    model,
    train_loader,
    epochs: int = 3,
    lr: float = 1e-5,
    device: str = "cuda",
) -> LogTemporalCritic:
    """
    Train the LogTemporalCritic using multi-layer attention supervision.

    Args:
        model: frozen HuggingFace causal LM (output_attentions=True)
        train_loader: DataLoader yielding {'input_ids': Tensor, 'labels': Tensor}
        epochs: number of training epochs
        lr: learning rate
        device: torch device string
    Returns:
        trained LogTemporalCritic
    """
    hidden_size = model.config.hidden_size
    critic = LogTemporalCritic(in_dim=hidden_size).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    logger.info(f"Training LogTemporalCritic on layers {TARGET_LAYERS} for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)

            # Always ensure output_attentions=True is passed
            with torch.no_grad():
                outs = model(ids, output_attentions=True, output_hidden_states=True)

            # Attention supervision target: how much each token is attended to
            # attn: [num_layers, batch, heads, seq, seq] → average over heads → [batch, seq, seq]
            attn_stack = torch.stack(
                [outs.attentions[i].mean(dim=1) for i in TARGET_LAYERS]
            )  # [num_target, batch, seq, seq]
            avg_attn = attn_stack.mean(dim=0).to(torch.float32)  # [batch, seq, seq]
            # Target: column sum (how much future tokens attend to each position)
            target = avg_attn.sum(dim=-1, keepdim=True)  # [batch, seq, 1]
            target = target / (target.max() + 1e-6)

            # Feature: block-level mean hidden state
            h_block = torch.stack([outs.hidden_states[i] for i in TARGET_LAYERS]).mean(dim=0)  # [batch, seq, dim]

            preds = critic(h_block)  # [batch, seq, 1]
            loss = F.mse_loss(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        logger.info(f"Critic epoch {epoch} | MSE loss: {avg:.6f}")
        if _WANDB and wandb.run is not None:
            wandb.log({"critic/epoch": epoch, "critic/mse_loss": avg})
        log_event("critic_epoch", {"epoch": epoch, "mse_loss": avg})

    log_event("critic_training_complete", {"epochs": epochs})
    return critic

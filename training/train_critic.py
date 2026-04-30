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
from utils.config import config

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_LAYERS = [10, 12, 14]


def _ensure_eager_attention(model):
    """Force eager attention so the model returns attention tensors."""
    if hasattr(model, "set_attn_implementation"):
        current_impl = getattr(model.config, "_attn_implementation", None)
        if current_impl != "eager":
            logger.info(
                "Switching attention backend from %s to eager for critic training",
                current_impl,
            )
            model.set_attn_implementation("eager")


def train_block_critic(
    model,
    train_loader,
    epochs: int = 3,
    lr: float = 1e-5,
    device: str = "cuda",
    target_layers: list = None,
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
    target_layers = target_layers if target_layers is not None else config.get("target_layers", _DEFAULT_TARGET_LAYERS)
    hidden_size = model.config.hidden_size
    critic = LogTemporalCritic(in_dim=hidden_size).to(device)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = getattr(train_loader, "batch_size", 1) or 1
    use_data_parallel = torch.cuda.is_available() and num_devices > 1 and batch_size >= num_devices

    if use_data_parallel:
        model = torch.nn.DataParallel(model)
        critic = torch.nn.DataParallel(critic)
    elif torch.cuda.is_available() and num_devices > 1:
        logger.warning(
            "Skipping DataParallel for critic training because batch_size=%s is smaller than num_gpus=%s.",
            batch_size,
            num_devices,
        )

    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    _ensure_eager_attention(base_model)

    logger.info(f"Training LogTemporalCritic on layers {target_layers} for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)

            model_to_call = model
            critic_to_call = critic
            if isinstance(model, torch.nn.DataParallel) and ids.size(0) < num_devices:
                model_to_call = model.module
            if isinstance(critic, torch.nn.DataParallel) and ids.size(0) < num_devices:
                critic_to_call = critic.module

            # Always ensure output_attentions=True is passed
            with torch.no_grad():
                outs = model_to_call(input_ids=ids, output_attentions=True, output_hidden_states=True)

            if not outs.attentions:
                raise RuntimeError(
                    "Model did not return attention tensors. " "Critic training requires eager attention backend."
                )

            # Attention supervision target: how much each token is attended to
            attn_stack = torch.stack(
                [outs.attentions[i].mean(dim=1) for i in target_layers]
            )  # [num_target, batch, seq, seq]
            avg_attn = attn_stack.mean(dim=0).to(torch.float32)  # [batch, seq, seq]
            target = avg_attn.sum(dim=-1)  # [batch, seq]
            target = target / (target.max() + 1e-6)

            # Feature: block-level mean hidden state
            h_block = torch.stack([outs.hidden_states[i] for i in target_layers]).mean(dim=0)  # [batch, seq, dim]

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                preds = critic_to_call(h_block)  # [batch, seq, 1]
                loss = F.mse_loss(preds, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        logger.info(f"Critic epoch {epoch} | MSE loss: {avg:.6f}")
        if _WANDB and wandb.run is not None:
            wandb.log({"critic/epoch": epoch, "critic/mse_loss": avg})
        log_event("critic_epoch", {"epoch": epoch, "mse_loss": avg})

    log_event("critic_training_complete", {"epochs": epochs})
    return critic.module if isinstance(critic, torch.nn.DataParallel) else critic

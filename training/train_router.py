"""
Router Training
================
Trains the SoftPlanningRouter (or legacy VoC Router) using
data-collected records from a forward pass with collect_data=True.

Pseudocode:
    for epoch in range(epochs):
        for record in records:
            pred = router(record.feat)
            loss = MSE(pred, record.label)
            loss.backward(); optimizer.step()
        if no improvement for `patience` epochs: early stop
"""
import torch
import logging
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
from logs.research_logger import log_event
from utils.config import config

logger = logging.getLogger(__name__)


def train_router(router, records: list, device: str = "cuda") -> object:
    """
    Train the router using collected (feature, importance label) records.

    Args:
        router: Router module (per_token or sequence-level)
        records: list of {'feat': Tensor, 'label': float}
        device: torch device string
    Returns:
        trained router
    """
    router.train()
    optimizer = torch.optim.Adam(router.parameters(), lr=config['learning_rate'])

    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(config['epochs']):
        total_loss = 0.0
        count = 0

        for record in records:
            feat = record["feat"].to(device)
            label = torch.tensor([record["label"]], device=device, dtype=torch.float32)

            if router.per_token:
                pred = router(feat).squeeze()
                label = label.expand_as(pred)
                loss = ((pred - label) ** 2).mean()
            else:
                pred = router(feat).squeeze()
                loss = (pred - label) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            break

        logger.info(f"Router epoch {epoch} | loss: {avg_loss:.4f}")
        if _WANDB and wandb.run is not None:
            wandb.log({"router/epoch": epoch, "router/loss": avg_loss})
        log_event("router_epoch", {"epoch": epoch, "loss": avg_loss})

    log_event("router_training_complete", {"best_loss": best_loss})
    return router

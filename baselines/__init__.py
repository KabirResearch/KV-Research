from baselines.static_skip import apply_static_skip
from baselines.random_skip import apply_random_skip
from baselines.token_pruning import apply_token_pruning
from baselines.early_exit import apply_early_exit
from baselines.moe import apply_moe
from baselines.mod import apply_mod
from baselines.speculative import speculative_decode

__all__ = [
    "apply_static_skip",
    "apply_random_skip",
    "apply_token_pruning",
    "apply_early_exit",
    "apply_moe",
    "apply_mod",
    "speculative_decode",
]

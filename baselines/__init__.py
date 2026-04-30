from baselines.static_skip import apply_static_skip
from baselines.random_skip import apply_random_skip
from baselines.mod import apply_mod
from baselines.token_pruning import apply_token_pruning
from baselines.speculative import speculative_decode

__all__ = [
    "apply_static_skip",
    "apply_random_skip",
    "apply_mod",
    "apply_token_pruning",
    "speculative_decode",
]

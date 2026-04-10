"""
Backward-compatibility shim — the flat models.py is superseded by the models/ package.
All imports from this file will continue to work, but prefer:
    from models.critics import LogTemporalCritic
    from models.router  import SoftPlanningRouter
    from models._legacy import VoCModel, Router, VoCSkipLayer, TokenLevelVoCSkipLayer
"""
from models import (  # noqa: F401
    LogTemporalCritic,
    SoftPlanningRouter,
    apply_skip,
    apply_entropy_skip,
    VoCModel,
    Router,
    VoCSkipLayer,
    TokenLevelVoCSkipLayer,
)

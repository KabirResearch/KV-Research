# Redirects to evaluation package. Kept for backward compatibility.
from evaluation.evaluate import evaluate_goldilocks, run_full, run_skip
from evaluation.calibrate import calibrate_voc_thresholds

# Legacy aliases
def run_entropy(*args, **kwargs):
    raise NotImplementedError("run_entropy is removed. Use evaluation.evaluate.run_skip with baselines.static_skip.")

def run_voc(*args, **kwargs):
    raise NotImplementedError("run_voc is removed. Use evaluation.evaluate.run_skip with models.router.")

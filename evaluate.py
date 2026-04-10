# Redirects to evaluation package. Kept for backward compatibility.


# Legacy aliases
def run_entropy(*args, **kwargs):
    raise NotImplementedError("run_entropy is removed. Use evaluation.evaluate.run_skip with baselines.static_skip.")


def run_voc(*args, **kwargs):
    raise NotImplementedError("run_voc is removed. Use evaluation.evaluate.run_skip with models.router.")

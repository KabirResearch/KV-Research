import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import config
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cache = {}


def load_model():
    model_name = config["model_name"]
    if model_name in model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return model_cache[model_name]
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).float().to(device)
    model.eval()
    if tokenizer.pad_token is None:
        logger.debug("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    # Patch model to always have model.gpt_neox.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        pass
    else:

        class _FakeNeoX:
            def __init__(self, layers):
                self.layers = layers

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.gpt_neox = _FakeNeoX(model.transformer.h)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.gpt_neox = _FakeNeoX(model.model.layers)
        elif hasattr(model, "layers"):
            model.gpt_neox = _FakeNeoX(model.layers)
        else:
            raise AttributeError("Model does not have a recognizable layers attribute for patching gpt_neox.layers.")
    model_cache[model_name] = (model, tokenizer)
    logger.info(f"Model loaded successfully: {model_name}")
    return model, tokenizer

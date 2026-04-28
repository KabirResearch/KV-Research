import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import config
import logging

logger = logging.getLogger(__name__)
model_cache = {}


def _resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if arch not in supported_arches:
            gpu_name = torch.cuda.get_device_name(0)
            logger.warning(
                "CUDA device %s (%s) is not supported by this PyTorch build. "
                "Falling back to CPU; use a T4/V100/A100 runtime or install a torch wheel that supports %s.",
                gpu_name,
                arch,
                arch,
            )
            return torch.device("cpu")
    except Exception as exc:
        logger.warning("Could not validate CUDA capability; falling back to CUDA device selection: %s", exc)

    return torch.device("cuda")


device = _resolve_device()


def load_model():
    model_name = config["model_name"]
    if model_name in model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return model_cache[model_name]
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs = {"dtype": model_dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)
    model.eval()
    logger.info("Loaded model on %s with dtype %s", device, model_dtype)
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

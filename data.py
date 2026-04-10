import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.config import config

logger = logging.getLogger(__name__)


def load_dataset_part(split=None):
    split = split or config["dataset_split"]
    logger.info(f"Loading dataset {config['dataset_name']} split {split}")
    dataset = load_dataset(config["dataset_name"], config["dataset_config"], split=split)
    logger.info(f"Loaded {len(dataset)} raw samples")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    logger.info(f"Filtered to {len(dataset)} valid samples")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=config["max_length"])

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset

"""
Dataset Loading
================
Loads and tokenizes WikiText-103 (or any HuggingFace dataset).

Critical fix: labels use -100 masking at padding positions.
Without this, PPL is inflated because padding tokens are included
in the cross-entropy loss denominator.

Pseudocode:
    dataset = load_dataset(name, config, split)
    dataset = filter(non-empty)
    for each example:
        tokens = tokenizer(text, truncation, padding, max_length)
        labels = tokens["input_ids"].clone()
        labels[labels == pad_token_id] = -100   ← CRITICAL
    dataset.set_format(["input_ids", "labels"])
"""
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.config import config

logger = logging.getLogger(__name__)


def load_dataset_part(split: str = None, min_text_length: int = 20):
    """
    Load, filter, and tokenize the configured dataset.
    Labels are -100 masked at padding positions for correct PPL.

    Args:
        split: dataset split override (default from config)
        min_text_length: minimum character length to filter noise
    Returns:
        HuggingFace Dataset with columns: input_ids, labels
    """
    split = split or config['dataset_split']
    logger.info(f"Loading dataset: {config['dataset_name']} / {config['dataset_config']} [{split}]")

    dataset = load_dataset(config['dataset_name'], config['dataset_config'], split=split)
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > min_text_length)
    logger.info(f"Filtered {before} → {len(dataset)} samples (min_len={min_text_length})")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        outputs = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config['max_length'],
        )
        # -100 mask: padding tokens are ignored in loss computation
        labels = [
            [-100 if t == tokenizer.pad_token_id else t for t in ids]
            for ids in outputs["input_ids"]
        ]
        outputs["labels"] = labels
        return outputs

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    logger.info(f"Dataset ready: {len(dataset)} samples, max_length={config['max_length']}")
    return dataset

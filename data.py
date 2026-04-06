import json
from datasets import load_dataset
from transformers import AutoTokenizer

with open('config.json') as f:
    config = json.load(f)

def load_dataset_part():
    dataset = load_dataset(config['dataset_name'], config['dataset_config'], split=config['dataset_split'])
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=config['max_length'])
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset
import json
import os

config_file = os.environ.get("CONFIG_FILE", "config.json")
with open(config_file) as f:
    config = json.load(f)

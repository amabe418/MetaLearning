import yaml
from pathlib import Path

def load_config(file_name: str):
    path = Path(__file__).parent / file_name
    with open(path, "r") as f:
        return yaml.safe_load(f)

datasets = load_config("datasets.yaml")
models = load_config("models.yaml")
benchmarks = load_config("benchmarks.yaml")["benchmarks"]
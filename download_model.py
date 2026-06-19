from pathlib import Path
from huggingface_hub import snapshot_download
import argparse

MODELS = {
    "qwen3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "patterns": [
            "qwen2.5-3b-instruct-q4_k_m.gguf",
        ],
    },
    "qwen7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "patterns": [
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["qwen3b", "qwen7b"],
    required=True,
)
args = parser.parse_args()

model = MODELS[args.model]

path = Path("/workspace/models")
path.mkdir(parents=True, exist_ok=True)

already_downloaded = all(
    (path / p).exists()
    for p in model["patterns"]
)

if already_downloaded:
    print(f"{args.model} already exists.")
else:
    snapshot_download(
        repo_id=model["repo"],
        allow_patterns=model["patterns"],
        local_dir=str(path),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded {args.model} to {path}")
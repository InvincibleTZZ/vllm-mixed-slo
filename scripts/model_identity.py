"""Fingerprint local HF weights and tokenizer files without loading model tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def model_identity(root):
    root = Path(root)
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("the model directory has no Safetensors or PyTorch weight files")
    metadata = ("config.json", "generation_config.json", "model.safetensors.index.json",
                "pytorch_model.bin.index.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
                "spiece.model", "vocab.json", "merges.txt")
    paths = weights + [root / name for name in metadata if (root / name).is_file()]
    files = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        files[path.name] = digest.hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(model_identity(args.model), indent=2))

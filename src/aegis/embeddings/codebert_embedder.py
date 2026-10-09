"""CodeBERT-based semantic embeddings for source code functions.

Loads microsoft/codebert-base once and embeds arbitrary source text using
the [CLS] token's final hidden state as a fixed 768-dimension vector.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/codebert-base"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str | None = None):
    """Loads CodeBERT onto `device` (default: the GPU if one is available)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device or default_device())
    model.eval()
    return tokenizer, model


def embed(tokenizer, model, source_text: str) -> tuple[list[float], int]:
    """Embed one source-text snippet. Returns (768-dim vector, token_count)."""
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding[0].tolist(), inputs["input_ids"].shape[1]


def embed_files(paths: list[Path], project_root: Path, output_dir: Path) -> list[dict]:
    """Embed a list of source files, writing one JSON per file plus a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_NAME} ...")
    tokenizer, model = load_model()

    manifest = []
    for path in paths:
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue

        lang = path.suffix.lstrip(".")
        source_text = path.read_text()
        vector, token_count = embed(tokenizer, model, source_text)

        out_path = output_dir / (path.stem + f"_{lang}_embedding.json")
        out_path.write_text(json.dumps({
            "file": str(path.relative_to(project_root)),
            "language": lang,
            "model": MODEL_NAME,
            "token_count": token_count,
            "embedding_dim": len(vector),
            "embedding": vector,
        }))

        manifest.append({
            "file": str(path.relative_to(project_root)),
            "language": lang,
            "token_count": token_count,
            "embedding_dim": len(vector),
            "output": str(out_path.relative_to(project_root)),
        })
        print(f"[{lang}] {path.name:30s} tokens={token_count:4d}  dim={len(vector)}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nEmbedded {len(manifest)} file(s). Manifest: {output_dir / 'manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed source files with CodeBERT.")
    parser.add_argument("files", type=Path, nargs="+", help="Source files to embed")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write embeddings + manifest")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                         help="Root used to compute relative file paths in the manifest")
    args = parser.parse_args()
    embed_files(args.files, args.project_root, args.output_dir)


if __name__ == "__main__":
    main()

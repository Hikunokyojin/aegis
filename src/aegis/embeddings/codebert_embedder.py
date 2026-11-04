"""CodeBERT-based semantic embeddings for source code functions.

Loads microsoft/codebert-base once and embeds arbitrary source text using
the [CLS] token's final hidden state as a fixed 768-dimension vector.
"""
import torch
from transformers import AutoModel, AutoTokenizer

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

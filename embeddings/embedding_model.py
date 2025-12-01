# embeddings/embedding_model.py
import os
from typing import Iterable
from sentence_transformers import SentenceTransformer
import numpy as np

_EMBEDDER = None

def _as_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on", "y")

class EmbeddingModel:
    def __init__(self):
        # Stronger English embedder that fits 8 GB and needs no trust_remote_code
        self.model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        # Auto device; override via EMBED_DEVICE=cuda or cpu
        self.device = os.getenv("EMBED_DEVICE", None)
        # Normalize for cosine similarity (index and query must match)
        self.normalize = _as_bool(os.getenv("EMBED_NORMALIZE", "1"), default=True)
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

def get_embedding_model() -> EmbeddingModel:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = EmbeddingModel()
    return _EMBEDDER
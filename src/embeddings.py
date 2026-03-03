"""
Embedding model loading + embedding + alignment scoring.

"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import clean_up

def build_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,) -> np.ndarray:
    texts = [clean_up(t) for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

def compute_alignment_scores(embeddings: np.ndarray, aims_emb: np.ndarray) -> np.ndarray:
    """
    Backwards-compatible scoring vs a single aims embedding.
    Returns shape (N,).
    """
    if aims_emb.ndim == 1:
        aims_emb = aims_emb.reshape(1, -1)
    return cosine_similarity(embeddings, aims_emb).reshape(-1)

def compute_alignment_matrix(embeddings: np.ndarray, aims_embs: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix between paper embeddings and aims embeddings.

    embeddings: (N, D)
    aims_embs: (K, D) or (D,)

    Returns: (N, K)
    """
    if aims_embs.ndim == 1:
        aims_embs = aims_embs.reshape(1, -1)
    return cosine_similarity(embeddings, aims_embs)

def compute_alignment_scores_all(
    embeddings: np.ndarray,
    aims_embs: np.ndarray,
    *,
    topk: int = 3,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute alignment scores vs multiple aims-sentence embeddings using 3 aggregations.

    Returns:
    - score_max:        (N,)
    - best_idx:         (N,) argmax index into aims sentences (based on max similarity)
    - score_mean:       (N,)
    - score_topk_mean:  (N,)
    """
    sim = compute_alignment_matrix(embeddings, aims_embs)  # (N, K)

    best_idx = np.argmax(sim, axis=1)
    score_max = np.max(sim, axis=1)
    score_mean = np.mean(sim, axis=1)

    k = min(int(topk), sim.shape[1])
    if k < 1:
        raise ValueError("topk must be >= 1")
    topk_vals = np.partition(sim, -k, axis=1)[:, -k:]
    score_topk_mean = np.mean(topk_vals, axis=1)

    return score_max, best_idx, score_mean, score_topk_mean
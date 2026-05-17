"""
utils/embeddings.py — ResNet-50 embedding pipeline + FAISS index management
"""
from pathlib import Path

import numpy as np
import faiss
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import sys
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import (
    EMBED_DIM, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    FAISS_INDEX_SUFFIX, EMBED_NPY_SUFFIX, CACHE_DIR,
)


# ── Model ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str | None = None):
    """
    ResNet-50 with classifier head removed → 2048-d embedding extractor.
    Cached in Streamlit via @st.cache_resource.
    """
    if device is None:
        device = get_device()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device), device


# ── Embedding ─────────────────────────────────────────────────────────────────

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def embed_chips(
    chips: np.ndarray,
    model,
    device: str,
    batch_size: int = BATCH_SIZE,
    progress_callback=None,
) -> np.ndarray:
    """
    Embed (N, 3, H, W) float32 chips → (N, EMBED_DIM) L2-normalized float32.

    Parameters
    ----------
    progress_callback : callable(done, total) | None
        Called after each batch for progress reporting.
    """
    all_embs = []
    total = len(chips)

    for start in range(0, total, batch_size):
        batch_np = chips[start : start + batch_size]
        batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
        batch = torch.stack([_normalize(b) for b in batch])

        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        all_embs.append(emb)

        if progress_callback is not None:
            progress_callback(min(start + batch_size, total), total)

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    faiss.normalize_L2(embs)
    return embs


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """Cosine similarity index (inner product on L2-normalized vectors)."""
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def query_index(
    index: faiss.IndexFlatIP,
    embs: np.ndarray,
    query_idx: int,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """
    Find top_k most similar chips to query_idx, excluding itself.

    Returns
    -------
    result_indices : list[int]
    result_scores  : list[float]  cosine similarities in [−1, 1]
    """
    query_vec = embs[query_idx : query_idx + 1]
    # Fetch top_k+1 to guarantee we can exclude the query itself
    distances, indices = index.search(query_vec, top_k + 1)

    result_indices, result_scores = [], []
    for idx, score in zip(indices[0], distances[0]):
        if idx == query_idx:
            continue
        result_indices.append(int(idx))
        result_scores.append(float(score))
        if len(result_indices) == top_k:
            break

    return result_indices, result_scores


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(index: faiss.IndexFlatIP, path: Path):
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


def save_embeddings(embs: np.ndarray, path: Path):
    np.save(str(path), embs)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(str(path))


# ── UMAP projection ───────────────────────────────────────────────────────────

def umap_project(embs: np.ndarray, n_components: int = 2, n_neighbors: int = 15):
    """
    2-D UMAP projection of embedding matrix.
    Returns (N, 2) float32 array. Lazy import so UMAP is optional.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    return reducer.fit_transform(embs).astype(np.float32)

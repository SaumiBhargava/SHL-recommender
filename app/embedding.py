# ============================================================
# embedding.py
# ============================================================
# Shared text -> vector embedder used by BOTH build_index.py and
# retrieval.py, so the saved index and the live query embeddings come
# from an identical pipeline.
#
# WHY fastembed INSTEAD OF sentence-transformers?
# -----------------------------------------------
# It runs the SAME model (all-MiniLM-L6-v2, 384 dims) but through ONNX
# Runtime instead of PyTorch. PyTorch alone needs ~300-500MB of RAM just
# to load, which blows past Render's free 512MB tier. ONNX Runtime is a
# fraction of that, so the service fits on the free plan. Same weights ->
# same embeddings (to ~5 decimal places) -> same search quality.
# ============================================================

import numpy as np
import faiss
from fastembed import TextEmbedding

# fastembed identifies the model by its full Hugging Face name.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Loaded lazily and cached so we only build it once per process.
_model = None


def get_embedder() -> TextEmbedding:
    """Return the cached embedder, creating it on first use."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _model


def embed_texts(texts) -> np.ndarray:
    """
    Embed a list of strings.

    Returns an (n, 384) float32 numpy array, L2-normalized so that an
    inner-product search (faiss.IndexFlatIP) behaves as cosine similarity.
    Works for a single-item list too -> shape (1, 384).
    """
    model = get_embedder()
    vectors = np.array(list(model.embed(list(texts))), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    # Normalize explicitly so we never depend on the backend's defaults.
    faiss.normalize_L2(vectors)
    return vectors

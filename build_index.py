# ============================================================
# build_index.py
# ============================================================
# YOU RUN THIS ONCE LOCALLY before deploying.
# It reads your shl_product_catalog.json and creates two files:
#   - data/catalog.faiss    → the searchable vector index
#   - data/catalog_meta.json → maps index positions to assessment data
#
# WHAT IS A VECTOR INDEX?
# ------------------------
# Imagine each assessment as a point in space.
# Similar assessments (like "Java test" and "Python test") are
# placed close together. "OPQ32r personality" is far from "Java test".
#
# When a user says "I need to hire a software developer",
# we convert that phrase into a point in the same space,
# then find the 20 closest assessments to that point.
# That's semantic search — finding meaning, not just keywords.
#
# HOW WE MAKE THE VECTORS:
# -------------------------
# We use sentence-transformers (specifically "all-MiniLM-L6-v2").
# This model reads a string like "personality assessment for leadership"
# and outputs a list of 384 numbers (a "vector" or "embedding").
# Two strings with similar meaning → similar vectors → close in space.
#
# HOW FAISS STORES AND SEARCHES THEM:
# -------------------------------------
# FAISS (Facebook AI Similarity Search) stores all our vectors
# and can find the closest ones to a query vector in <10ms
# even with thousands of assessments. It's a file on disk — no server.
# ============================================================

import json
import os
import sys
import numpy as np
# numpy is Python's numerical computing library.
# We use it to create arrays of numbers (our vectors).
# np.array([...]) creates an array.
# np.float32 is a data type — 32-bit floating point numbers.
# FAISS requires float32 specifically.

import faiss
# faiss is Facebook's similarity search library.
# faiss.IndexFlatIP → stores vectors, searches by inner product (cosine-like)
# index.add(vectors) → adds vectors to the index
# index.search(query, k) → finds k nearest vectors to query

from sentence_transformers import SentenceTransformer
# SentenceTransformer loads a pre-trained model that converts
# text strings into vectors (embeddings).
# model.encode(["text1", "text2"]) → returns array of vectors

# Add parent directory to path so we can import from app/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.config import (
    CATALOG_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    keys_to_test_type,
)


# ============================================================
# STEP 1: LOAD AND CLEAN THE CATALOG
# ============================================================

def load_catalog(path: str) -> list:
    """
    Loads shl_product_catalog.json and returns a cleaned list of assessments.
    
    Your JSON might be a list directly, or it might be wrapped in
    a dictionary. We handle both cases here.
    """
    print(f"Loading catalog from: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # json.load(f) reads the file and converts JSON → Python objects
    # A JSON array  → Python list
    # A JSON object → Python dict
    
    # Handle both {"results": [...]} and [...] formats
    if isinstance(raw, dict):
        # Try common wrapper keys
        for key in ("results", "data", "assessments", "products"):
            if key in raw:
                raw = raw[key]
                break
        else:
            # If none of the keys match, try to find any list value
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
    
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected shl_product_catalog.json to contain a list of assessments, "
            f"got {type(raw).__name__}. Check your JSON structure."
        )
    
    print(f"  Loaded {len(raw)} raw catalog entries")
    return raw


def clean_assessment(item: dict) -> dict | None:
    """
    Takes one raw catalog item and returns a clean, standardized dict.
    Returns None if the item is missing critical fields (name or link).
    
    Your catalog field names: name, link, keys, description,
    job_levels, remote, adaptive, duration, languages
    
    We keep all fields but standardize the key structure.
    """
    # Extract the fields we care about.
    # .get("field", default) safely gets a value — returns default if missing.
    name = (item.get("name") or "").strip()
    url  = (item.get("link") or "").strip()
    
    # Skip items without a name or URL — we can't recommend them
    if not name or not url:
        return None
    
    # Make sure the URL is an SHL URL
    if not url.startswith("https://www.shl.com"):
        return None
    
    # Convert the keys list to test_type letters
    keys = item.get("keys") or []
    test_type = keys_to_test_type(keys)
    
    # Build the clean assessment dict
    return {
        "name":        name,
        "url":         url,
        "test_type":   test_type,       # e.g. "K,S" or "P"
        "keys":        keys,            # e.g. ["Knowledge & Skills", "Simulations"]
        "description": (item.get("description") or "").strip(),
        "job_levels":  item.get("job_levels") or [],
        "remote":      item.get("remote", ""),       # "yes" / "no" / ""
        "adaptive":    item.get("adaptive", ""),     # "yes" / "no" / ""
        "duration":    (item.get("duration") or "").strip(),
        "duration_raw":(item.get("duration_raw") or "").strip(),
        "languages":   item.get("languages") or [],
    }


# ============================================================
# STEP 2: BUILD THE TEXT WE EMBED FOR EACH ASSESSMENT
# ============================================================

def build_embedding_text(assessment: dict) -> str:
    """
    Creates the text string we embed for each assessment.
    This string represents everything meaningful about the assessment.
    
    WHY THIS MATTERS:
    When a user says "I need cognitive tests for graduates",
    we embed that query and compare it to our assessment embeddings.
    If we only embedded the name, "Verify G+" might not match.
    But if we embed name + description + job_levels + keys,
    "cognitive ability graduate aptitude" will be close to the query.
    
    The richer the text, the better the semantic search.
    """
    parts = []
    
    # Always include the name (most important)
    if assessment["name"]:
        parts.append(assessment["name"])
    
    # Include the human-readable key names (very useful for matching)
    # e.g. "Personality & Behavior Knowledge & Skills"
    if assessment["keys"]:
        parts.append(" ".join(assessment["keys"]))
    
    # Include description (contains the most semantic information)
    if assessment["description"]:
        # Limit to first 300 chars to avoid very long embeddings
        # (longer text = diminishing returns for short queries)
        parts.append(assessment["description"][:300])
    
    # Include job levels (helps match "senior" or "graduate" queries)
    if assessment["job_levels"]:
        parts.append(" ".join(assessment["job_levels"]))
    
    # Include remote/adaptive info (helps with constraint-based queries)
    if assessment["remote"] == "yes":
        parts.append("remote testing available")
    if assessment["adaptive"] == "yes":
        parts.append("adaptive IRT")
    
    # Join all parts with a space separator
    return " ".join(parts)


# ============================================================
# STEP 3: CREATE AND SAVE THE FAISS INDEX
# ============================================================

def build_index(assessments: list, model: SentenceTransformer) -> tuple:
    """
    Takes cleaned assessments, generates embeddings, builds FAISS index.
    
    Returns (index, metadata) where:
      index    → the FAISS index object (for saving to .faiss file)
      metadata → list of assessment dicts (same order as index rows)
    
    IMPORTANT: metadata[i] corresponds to index row i.
    When FAISS says "row 3 is the closest match",
    we look up metadata[3] to get the actual assessment data.
    """
    print("Building embedding texts...")
    texts = [build_embedding_text(a) for a in assessments]
    # texts[i] corresponds to assessments[i]
    
    print(f"Encoding {len(texts)} assessments with sentence-transformers...")
    print("  (This takes 1-3 minutes the first time — model downloads ~90MB)")
    
    # encode() converts each text string to a 384-dimensional vector.
    # show_progress_bar=True prints a progress bar.
    # convert_to_numpy=True gives us numpy arrays (required by FAISS).
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,    # process 32 at a time (memory efficient)
    )
    # embeddings.shape == (num_assessments, 384)
    # Each row is one assessment's 384-dimensional vector
    
    # Normalize vectors to unit length (required for cosine similarity)
    # Without normalization, inner product ≠ cosine similarity.
    # faiss.normalize_L2 modifies the array in-place.
    faiss.normalize_L2(embeddings)
    
    # Get the dimension of our vectors (384 for MiniLM)
    dimension = embeddings.shape[1]
    # embeddings.shape returns (rows, cols) — we want cols (the vector size)
    
    print(f"  Vector dimension: {dimension}")
    
    # Create a FAISS index.
    # IndexFlatIP = Flat Index using Inner Product similarity
    # "Flat" means it stores all vectors exactly (no compression).
    # "IP" (Inner Product) = dot product, which equals cosine similarity
    # when vectors are normalized (which we did above).
    index = faiss.IndexFlatIP(dimension)
    
    # Convert to float32 — FAISS requires this specific data type.
    embeddings_f32 = embeddings.astype(np.float32)
    
    # Add all vectors to the index.
    # After this, the index contains all our assessment vectors.
    index.add(embeddings_f32)
    
    print(f"  FAISS index built: {index.ntotal} vectors stored")
    
    return index, assessments


# ============================================================
# STEP 4: SAVE TO DISK
# ============================================================

def save_index(index, metadata: list, index_path: str, meta_path: str):
    """
    Saves the FAISS index and metadata to disk.
    
    index_path → binary .faiss file (FAISS's own format)
    meta_path  → JSON file with assessment data (name, url, etc.)
    
    WHY SAVE SEPARATELY?
    FAISS only stores numbers (the vectors). It doesn't know what
    "row 42" means — it just knows it's a vector. We store the
    actual assessment data in metadata.json, keyed by position.
    At query time: FAISS gives us row numbers → we look up metadata.
    """
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    # exist_ok=True means: don't raise an error if the dir already exists
    
    # Save the FAISS index to a binary file
    faiss.write_index(index, index_path)
    print(f"  Saved FAISS index to: {index_path}")
    
    # Save the metadata as JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    # ensure_ascii=False preserves non-ASCII characters (e.g. accented names)
    # indent=2 makes the JSON human-readable (for debugging)
    print(f"  Saved metadata to: {meta_path}")


# ============================================================
# MAIN FUNCTION — RUNS WHEN YOU EXECUTE THIS SCRIPT
# ============================================================

def main():
    print("=" * 60)
    print("SHL Catalog Index Builder")
    print("=" * 60)
    
    # --- Step 1: Load catalog ---
    raw_items = load_catalog(CATALOG_PATH)
    
    # --- Step 2: Clean items ---
    print("Cleaning catalog items...")
    assessments = []
    skipped = 0
    for item in raw_items:
        cleaned = clean_assessment(item)
        if cleaned:
            assessments.append(cleaned)
        else:
            skipped += 1
    
    print(f"  {len(assessments)} valid assessments")
    print(f"  {skipped} skipped (missing name or URL)")
    
    if len(assessments) == 0:
        print("ERROR: No valid assessments found. Check your shl_product_catalog.json.")
        sys.exit(1)
    
    # --- Step 3: Load embedding model ---
    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # all-MiniLM-L6-v2 is a well-balanced model:
    #   - Small: ~90MB download, loads in ~2 seconds
    #   - Fast: ~80ms per batch of 32 texts
    #   - Good quality: strong semantic understanding
    # It downloads automatically the first time, then caches locally.
    print("  Model loaded")
    
    # --- Step 4: Build FAISS index ---
    index, metadata = build_index(assessments, model)
    
    # --- Step 5: Save to disk ---
    print("Saving index to disk...")
    save_index(index, metadata, FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    
    # --- Done ---
    print()
    print("=" * 60)
    print("Index built successfully!")
    print(f"  Assessments indexed: {len(metadata)}")
    print(f"  FAISS file: {FAISS_INDEX_PATH}")
    print(f"  Metadata file: {FAISS_METADATA_PATH}")
    print()
    print("Next step: python -m uvicorn app.main:app --reload")
    print("=" * 60)


# This block runs only when you execute this file directly:
#   python build_index.py
# It does NOT run when another file imports from this module.
if __name__ == "__main__":
    main()
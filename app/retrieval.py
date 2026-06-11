# ============================================================
# retrieval.py
# ============================================================
# This file handles everything related to SEARCHING the catalog.
#
# It does three things:
#   1. Loads the FAISS index + metadata into memory at startup
#   2. Converts a text query into a vector and searches FAISS
#   3. Returns matching assessment dicts with all their fields
#
# IMPORTANT: This module is loaded ONCE when the server starts.
# The model and index stay in memory — we never reload them per request.
# That's why the /health endpoint can be slow on first call (loading)
# but /chat is fast after that.
# ============================================================

import json
import os
import numpy as np
import faiss
from typing import List, Optional
import logging

# Set up logging so we can see what's happening without print statements.
# logging is better than print in production — can be configured to write
# to files, have different severity levels (INFO, WARNING, ERROR), etc.
logger = logging.getLogger(__name__)
# __name__ is the module name ("retrieval") — keeps log messages organized.

# Import our config values
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RETRIEVAL_TOP_K,
)
from app.embedding import get_embedder, embed_texts


# ============================================================
# THE RETRIEVER CLASS
# ============================================================
# We use a class (not just functions) because we need to load
# the model and index ONCE and reuse them across many requests.
# A class lets us store them as instance variables (self.model, etc.)
# and call search() without reloading anything.

class CatalogRetriever:
    """
    Loads the FAISS index and sentence-transformer model once,
    then answers semantic search queries in milliseconds.
    
    Usage:
        retriever = CatalogRetriever()   # loads model + index
        results = retriever.search("cognitive test for graduate hiring", k=10)
        # returns list of assessment dicts
    """

    def __init__(self):
        # These will be set in load(). Using None as initial value
        # makes it clear they haven't been loaded yet.
        self.model    = None   # SentenceTransformer model
        self.index    = None   # FAISS index
        self.metadata = []     # list of assessment dicts (same order as index)
        self.loaded   = False  # flag so we know if loading succeeded
        
        # Also build a URL→assessment lookup for fast URL validation
        # and for the "compare" behavior (fetch specific assessments by URL)
        self.url_to_assessment = {}  # {url_string: assessment_dict}

    def load(self) -> None:
        """
        Loads the sentence-transformer model, FAISS index, and metadata.
        Call this once at server startup.
        
        Raises RuntimeError if files don't exist (user forgot to run build_index.py).
        """
        # --- Load the embedding model (ONNX via fastembed) ---
        logger.info("Loading embedding model (fastembed / all-MiniLM-L6-v2)...")
        self.model = get_embedder()
        # On first run: downloads the small ONNX model and caches locally.
        # On subsequent runs: loads from cache in ~1 second. Far less RAM
        # than the PyTorch version, so it fits Render's free tier.
        logger.info("  Model loaded")

        # --- Load the FAISS index ---
        if not os.path.exists(FAISS_INDEX_PATH):
            raise RuntimeError(
                f"FAISS index not found at: {FAISS_INDEX_PATH}\n"
                "Run 'python build_index.py' first to generate it."
            )
        logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info(f"  Index loaded: {self.index.ntotal} vectors")

        # --- Load the metadata ---
        if not os.path.exists(FAISS_METADATA_PATH):
            raise RuntimeError(
                f"Metadata file not found at: {FAISS_METADATA_PATH}\n"
                "Run 'python build_index.py' first to generate it."
            )
        logger.info(f"Loading metadata from: {FAISS_METADATA_PATH}")
        with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"  Metadata loaded: {len(self.metadata)} assessments")

        # --- Build URL lookup dict ---
        # This lets us instantly find an assessment by its URL.
        # Used for: comparison requests, URL whitelist validation.
        for assessment in self.metadata:
            url = assessment.get("url", "")
            if url:
                self.url_to_assessment[url] = assessment

        self.loaded = True
        logger.info("CatalogRetriever ready.")

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Converts a text query into a normalized vector.
        
        This is the same process we did for each assessment in build_index.py.
        When query vector is close to an assessment vector → good match.
        
        Returns a numpy array of shape (1, 384) — FAISS expects 2D arrays.
        """
        # embed_texts returns shape (1, 384), float32, already L2-normalized
        # (same pipeline build_index.py uses, so query and index vectors match).
        return embed_texts([query])

    def search(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
        filter_test_types: Optional[List[str]] = None,
        filter_remote: Optional[bool] = None,
    ) -> List[dict]:
        """
        Searches the catalog for assessments matching the query.
        
        Args:
            query           : natural language search string
                              e.g. "personality leadership senior executive"
            k               : number of results to return (default: from config)
            filter_test_types: if set, only return assessments with these type letters
                              e.g. ["P", "A"] for Personality + Ability
            filter_remote   : if True, only return remote-testing assessments
        
        Returns:
            List of assessment dicts (from metadata), ordered by relevance.
        
        Example:
            results = retriever.search("Java developer skills test", k=5)
            # returns: [{"name": "Java 8 (New)", "url": "...", ...}, ...]
        """
        if not self.loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        
        if not query or not query.strip():
            return []
        
        # Step 1: Embed the query
        query_vector = self._embed_query(query.strip())
        # query_vector.shape == (1, 384)
        
        # Step 2: Search FAISS
        # We ask for more results than needed so we have room to filter.
        # If k=10 and we filter, we might end up with fewer than 10.
        # So we ask for k*3 first, then filter down to k.
        search_k = min(k * 3, self.index.ntotal)
        # min() prevents asking for more results than we have assessments
        
        distances, indices = self.index.search(query_vector, search_k)
        # distances.shape == (1, search_k) — similarity scores (-1 to 1)
        # indices.shape   == (1, search_k) — positions in metadata list
        # indices[0] is the first (and only) row since we searched one query
        
        # Step 3: Build result list from metadata
        results = []
        for idx in indices[0]:
            # indices can contain -1 if FAISS has fewer results than requested
            if idx == -1:
                continue
            
            assessment = self.metadata[idx]  # look up the assessment dict
            
            # Step 4: Apply filters
            
            # Filter by test type if requested
            if filter_test_types:
                # assessment["test_type"] is like "K,S" or "P"
                # We split by comma and check for intersection
                assessment_types = set(
                    t.strip() for t in assessment.get("test_type", "").split(",")
                )
                # If none of the requested types are in this assessment's types, skip it
                if not assessment_types.intersection(set(filter_test_types)):
                    continue
            
            # Filter by remote testing if requested
            if filter_remote is True:
                if assessment.get("remote", "").lower() != "yes":
                    continue
            
            results.append(assessment)
            
            # Stop once we have enough results
            if len(results) >= k:
                break
        
        return results

    def get_by_url(self, url: str) -> Optional[dict]:
        """
        Returns the assessment dict for a given URL, or None if not found.
        
        Used for:
        - Comparison requests: user asks "compare X and Y by name/url"
        - URL validation: checking that a URL the LLM returned is real
        
        Example:
            assessment = retriever.get_by_url(
                "https://www.shl.com/products/product-catalog/view/java-8-new/"
            )
        """
        return self.url_to_assessment.get(url)

    def get_by_name(self, name: str) -> Optional[dict]:
        """
        Searches for an assessment by name (case-insensitive, partial match).
        
        Used for comparison requests where the user says a name like "OPQ32r"
        and we need to find the full catalog entry.
        
        Returns the best match or None.
        """
        name_lower = name.lower().strip()
        
        # First try exact match (case-insensitive)
        for assessment in self.metadata:
            if assessment["name"].lower() == name_lower:
                return assessment
        
        # Then try partial match (name is contained in assessment name)
        for assessment in self.metadata:
            if name_lower in assessment["name"].lower():
                return assessment
        
        # No match found
        return None

    def validate_and_filter_urls(self, recommendations: list) -> list:
        """
        Takes a list of recommendation dicts from the LLM and removes any
        whose URLs don't exist in the real catalog.
        
        This is the HALLUCINATION GUARD — it prevents the LLM from
        inventing URLs and us returning them to the evaluator.
        
        Each item in recommendations should have: name, url, test_type
        
        Returns only items where url is in our catalog.
        """
        valid = []
        for rec in recommendations:
            url = rec.get("url", "")
            if url in self.url_to_assessment:
                # URL is real — include this recommendation
                # But use the catalog's data for name and test_type
                # (the LLM might have gotten those slightly wrong)
                real = self.url_to_assessment[url]
                valid.append({
                    "name":      real["name"],          # use catalog name
                    "url":       real["url"],            # use catalog URL
                    "test_type": real["test_type"],      # use catalog test_type
                })
            else:
                # URL is hallucinated — drop it silently
                logger.warning(f"Dropping hallucinated URL: {url}")
        
        return valid

    def get_all_urls(self) -> set:
        """Returns the set of all valid URLs in the catalog. Used for validation."""
        return set(self.url_to_assessment.keys())

    @property
    def catalog_size(self) -> int:
        """Returns the number of assessments in the catalog."""
        return len(self.metadata)


# ============================================================
# SINGLETON INSTANCE
# ============================================================
# We create ONE instance of CatalogRetriever here.
# Every other file imports this single instance:
#   from app.retrieval import retriever
#
# WHY SINGLETON?
# The model (~90MB) and FAISS index are loaded into memory once.
# If we created a new instance per request, we'd reload 90MB every time.
# That would make every request take 15+ seconds.
#
# The singleton loads once at startup, stays in memory, serves all requests.

retriever = CatalogRetriever()
# Note: retriever.load() is NOT called here.
# It's called in main.py's startup event, so errors show up clearly
# in the server logs rather than at import time.
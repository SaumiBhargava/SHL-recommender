import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CATALOG_PATH: str = os.getenv(
    "CATALOG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "shl_product_catalog.json")
)

# Path to the FAISS index file we'll generate with build_index.py
FAISS_INDEX_PATH: str = os.getenv(
    "FAISS_INDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "catalog.faiss")
)

# Path to the metadata file (maps FAISS index positions → assessment data)
FAISS_METADATA_PATH: str = os.getenv(
    "FAISS_METADATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "catalog_meta.json")
)

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))

MAX_TURNS: int = int(os.getenv("MAX_TURNS", "8"))

FORCE_RECOMMEND_TURN: int = int(os.getenv("FORCE_RECOMMEND_TURN", "6"))


# ============================================================
# STARTUP VALIDATION
# ============================================================

def validate_config() -> None:
    """
    Call this once when the server starts to check everything is set.
    Raises a clear error message instead of a cryptic failure later.
    
    For example, if you forget to set GEMINI_API_KEY on Render,
    you'll get "GEMINI_API_KEY is not set" immediately at startup
    instead of a confusing error on the first chat request.
    """
    errors = []

    if not GEMINI_API_KEY:
        errors.append(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or Render environment variables. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    if not os.path.exists(CATALOG_PATH):
        errors.append(
            f"shl_product_catalog.json not found at {CATALOG_PATH}. "
            "Make sure your data/shl_product_catalog.json file is in the project."
        )

    if not os.path.exists(FAISS_INDEX_PATH):
        errors.append(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run:  python build_index.py  to generate it."
        )

    if errors:
        # Join all errors and raise one clear message.
        # This stops the server from starting with broken config.
        raise RuntimeError(
            "Configuration errors found:\n" +
            "\n".join(f"  • {e}" for e in errors)
        )


# ============================================================
# TEST KEYS TO TYPE MAPPING
# ============================================================
# Your catalog uses full-text "keys" like "Personality & Behavior".
# The API spec requires single-letter test_type codes like "P".
# This mapping converts between the two.
#
# Used in build_index.py when processing shl_product_catalog.json,
# and in retrieval.py when building recommendations.

KEYS_TO_TYPE: dict = {
    "Ability & Aptitude":            "A",
    "Assessment Exercises":          "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                  "C",
    "Development & 360":             "D",
    "Personality & Behavior":        "P",
    "Knowledge & Skills":            "K",
    "Simulations":                   "S",
    # fallback for any unexpected value
}

def keys_to_test_type(keys: list) -> str:
    """
    Converts a list of key strings to a comma-separated test_type string.
    
    Example:
        keys_to_test_type(["Knowledge & Skills", "Simulations"])
        → "K,S"
        
        keys_to_test_type(["Personality & Behavior"])
        → "P"
        
        keys_to_test_type([])
        → ""
    """
    if not keys:
        return ""
    
    letters = []
    for key in keys:
        # Strip whitespace from the key name, look it up in our mapping.
        letter = KEYS_TO_TYPE.get(key.strip())
        if letter and letter not in letters:
            # Only add if we recognize it, and avoid duplicates.
            letters.append(letter)
    
    # Join with comma: ["K", "S"] → "K,S"
    return ",".join(letters)

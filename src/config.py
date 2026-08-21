"""
Single source of configuration for the whole project — pipeline, generation,
evaluation and the Streamlit UI. There is deliberately only ONE config.py.
"""

import os
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
load_dotenv()

# Gemini API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Primary model with fallback isolation
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# ---- LLM provider (provider-agnostic: swap providers via .env only) ----
# Google Gemini via its OpenAI-compatible endpoint:
#   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or GEMINI_API_KEY
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
OPENAI_API_KEY = LLM_API_KEY

LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "true").lower() in ("1", "true", "yes")
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

# ---- Paths ----
BASE_DIR = _HERE                 # .../MansouraHack/src
PROJECT_ROOT = _PROJECT_ROOT     # .../MansouraHack

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
RAW_DOCS_DIR = os.path.join(ASSETS_DIR, "documents")
PARSED_DATA_DIR = os.path.join(ASSETS_DIR, "parsed")
CLEANED_DATA_DIR = os.path.join(ASSETS_DIR, "cleaned")

QDRANT_PATH = os.path.join(ASSETS_DIR, "qdrant")
COLLECTION_NAME = "who_diseases"

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
TESTSET_PATH = os.path.join(EVAL_DIR, "testset.json")
DEMO_TESTSET_PATH = os.path.join(EVAL_DIR, "testset_demo.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# ---- Ingestion ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

DOCUMENT_NAME = "WHO guidelines for clinical management of arboviral diseases (2025)"
SOURCE_URL = "https://iris.who.int/"

# Per-file overrides for multi-document ingestion. Any .md file in
# CLEANED_DATA_DIR that isn't listed here still gets ingested — it just falls
# back to (file stem, SOURCE_URL) so a new PDF is picked up automatically
# instead of being silently skipped.
DOCUMENT_NAMES: dict[str, tuple[str, str]] = {
    "MainFile.md": (DOCUMENT_NAME, SOURCE_URL),
    "ChikungunyaFile.md": (
        "WHO guidelines for clinical management of arboviral diseases (2025) — "
        "Chikungunya supplementary source",
        SOURCE_URL,
    ),
}

# ---- Retrieval ----
TOP_K = 5

# ---- Generation ----
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")
GENERATION_TEMPERATURE = 0.0

# ---- Evaluation ----
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")

# ---- App Metadata ----
APP_NAME = "VectorGuard"
APP_TAGLINE = "Evidence-based clinical information on mosquito-borne diseases."
APP_DISCLAIMER = (
    "VectorGuard provides educational information and does not replace "
    "professional medical advice, diagnosis, or treatment."
)

# Supported Medical Topics
PRIMARY_TOPICS = ["Dengue", "Zika", "Chikungunya", "Yellow Fever"]


if __name__ == "__main__":
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("BASE_DIR     :", BASE_DIR)
    print()
    for label, path in [
        ("assets", ASSETS_DIR),
        ("documents", RAW_DOCS_DIR),
        ("parsed", PARSED_DATA_DIR),
        ("cleaned", CLEANED_DATA_DIR),
        ("qdrant", QDRANT_PATH),
        ("evaluation", EVAL_DIR),
        ("testset.json", TESTSET_PATH),
        ("testset_demo.json", DEMO_TESTSET_PATH),
        ("style.css", os.path.join(ASSETS_DIR, "style.css")),
    ]:
        mark = "OK     " if os.path.exists(path) else "MISSING"
        print(f"[{mark}] {label:<18} {path}")
    print()
    print("GEMINI_API_KEY      :", "OK" if GEMINI_API_KEY else "MISSING")
    print("LLM_API_KEY         :", "OK" if LLM_API_KEY else "MISSING")
    print("GEMINI_MODEL        :", GEMINI_MODEL)
    print("GENERATION_MODEL    :", GENERATION_MODEL)
    print("JUDGE_MODEL         :", JUDGE_MODEL)
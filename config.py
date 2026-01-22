# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- env ----------
load_dotenv()

def _get(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default

# ---------- project paths ----------
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
STORAGE_DIR   = ROOT_DIR / "storage"
LOGS_DIR      = ROOT_DIR / "logs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR   = "storage/index"
MEETINGS_DIR= "storage/meetings"
# Persisted meetings + vector index
MEETINGS_DIR  = Path(_get("MEETINGS_DIR", str(STORAGE_DIR / "meetings")))
INDEX_DIR     = Path(_get("INDEX_DIR",    str(STORAGE_DIR / "index")))

# LoRA adapter output
LORA_DIR      = Path(_get("LORA_DIR",     str(ARTIFACTS_DIR / "lora_minutes")))

# Optional: training files
TRAIN_JSONL   = Path(_get("TRAIN_JSONL",  str(DATA_DIR / "lora_minutes_train.jsonl")))
VAL_JSONL     = Path(_get("VAL_JSONL",    str(DATA_DIR / "lora_minutes_val.jsonl")))

# Logs
LOG_TRAIN_DIR = LOGS_DIR / "training"
LOG_APP_DIR   = LOGS_DIR / "app"
LOG_CHAT_DIR  = LOGS_DIR / "chat"

# Create dirs on import
for p in [
    ARTIFACTS_DIR, STORAGE_DIR, LOGS_DIR,
    MEETINGS_DIR, INDEX_DIR, LORA_DIR,
    LOG_TRAIN_DIR, LOG_APP_DIR, LOG_CHAT_DIR
]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- models ----------
# Summarizer (base) and LoRA adapter location
MODEL_NAME     = _get("MODEL_NAME", "google/flan-t5-base")
# Whisper ASR model (base/small/medium/large or faster-whisper names if you swap library)
WHISPER_MODEL  = _get("WHISPER_MODEL", "base")
# RAG embedding model
EMBED_MODEL    = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Local chat model served by Ollama (ensure you've pulled it)
OLLAMA_MODEL   = _get("OLLAMA_MODEL", "llama3.1:8b")

# ---------- generation defaults ----------
# Minutes generation (FLAN-T5 + LoRA)
SUM_MAX_INPUT     = int(_get("SUM_MAX_INPUT", "1024"))
SUM_MAX_NEW       = int(_get("SUM_MAX_NEW",   "600"))
SUM_NUM_BEAMS     = int(_get("SUM_NUM_BEAMS", "4"))
SUM_TEMPERATURE   = float(_get("SUM_TEMPERATURE", "0.0"))

# ---------- RAG chunking/search ----------
CHUNK_SIZE        = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP     = int(_get("CHUNK_OVERLAP", "80"))
QA_TOPK           = int(_get("QA_TOPK", "8"))
STRICT_FALLBACK   = _get("STRICT_FALLBACK", "Not discussed in meetings.")

# ---------- utility ----------
def get_device() -> str:
    """Return 'cuda' | 'mps' | 'cpu' without hard-depending on torch at import time."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()

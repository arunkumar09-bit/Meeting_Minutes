# rag/store.py
import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# Logging
# ---------------------------
LOG = logging.getLogger("rag.store")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOG.addHandler(ch)

# ---------------------------
# Config / Paths
# ---------------------------
DEVICE = os.getenv("RAG_DEVICE", "mps")  # 'cuda' | 'mps' | 'cpu'
MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "storage/index")
EMB_PATH  = os.path.join(INDEX_DIR, "embeddings.npy")   # float32 (N, D)
TXT_PATH  = os.path.join(INDEX_DIR, "texts.jsonl")      # one JSON per line
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")       # one JSON per line
CFG_PATH  = os.path.join(INDEX_DIR, "config.json")      # store model/config used

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------
# Simple Document wrapper (to match your chat code)
# ---------------------------
@dataclass
class Document:
    page_content: str
    metadata: Dict

# ---------------------------
# Model load (singleton)
# ---------------------------
_model = None
def get_model():
    global _model
    if _model is None:
        t0 = time.time()
        LOG.info("Use pytorch device_name: %s", DEVICE)
        LOG.info("Load pretrained SentenceTransformer: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump({"model": MODEL_NAME, "device": DEVICE, "time": time.time()}, f, indent=2)
        LOG.info("embed model ready in %.2fs", time.time() - t0)
    return _model

# ---------------------------
# Index persistence helpers
# ---------------------------
def _load_lines(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _append_lines(path: str, records: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _load_embs() -> np.ndarray:
    if not os.path.exists(EMB_PATH):
        return np.zeros((0, 384), dtype=np.float32)  # default dim for MiniLM; will be overwritten on first write
    return np.load(EMB_PATH)

def _save_embs(embs: np.ndarray):
    np.save(EMB_PATH, embs)

# ---------------------------
# Text chunking / embedding
# ---------------------------
def _chunk_text(s: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    chunks, i, n = [], 0, len(s)
    while i < n:
        end = min(n, i + max_chars)
        chunks.append(s[i:end])
        if end >= n:  # last chunk
            break
        i = max(0, end - overlap)
    return chunks

def _embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    m = get_model()
    t0 = time.time()
    LOG.info("embed: encoding %d chunks (batch_size=%d)", len(texts), batch_size)
    embs = m.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    LOG.info("embed: done in %.2fs", time.time() - t0)
    return embs

# ---------------------------
# Upsert meeting
# ---------------------------
def upsert_meeting(transcript: str, minutes: str, meta: Dict):
    """
    Splits transcript/minutes into chunks, embeds them, and appends to the on-disk index.
    meta should include at least: title, date, meeting_id
    """
    t0 = time.time()
    title = meta.get("title", "")
    date  = meta.get("date", "")
    mid   = meta.get("meeting_id", "")
    LOG.info("upsert: start | title=%s date=%s meeting_id=%s", title, date, mid)

    # 1) chunk
    tr_chunks = _chunk_text(transcript, max_chars=1200, overlap=150)
    mm_chunks = _chunk_text(minutes,    max_chars=1200, overlap=150)
    LOG.info("upsert: chunks | transcript=%d minutes=%d", len(tr_chunks), len(mm_chunks))

    # 2) build text+metadata rows (aligned)
    texts, metas = [], []
    for c in tr_chunks:
        texts.append(c)
        metas.append({"source": "transcript", **meta})
    for c in mm_chunks:
        texts.append(c)
        metas.append({"source": "minutes", **meta})

    if not texts:
        LOG.warning("upsert: no text to index; skipping.")
        return

    # 3) embed
    vecs = _embed_texts(texts, batch_size=8)  # small batch safe for CPU/MPS

    # 4) load current index, append, save
    old = _load_embs()
    new = vecs if old.size == 0 else np.concatenate([old, vecs], axis=0)
    _save_embs(new)

    # 5) persist texts & metas
    _append_lines(TXT_PATH, [{"text": t} for t in texts])
    _append_lines(META_PATH, metas)

    LOG.info("upsert: write done | added=%d | total=%d | time=%.2fs", len(texts), new.shape[0], time.time() - t0)

# ---------------------------
# Search & rerank
# ---------------------------
def _load_index() -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    embs = _load_embs()
    texts = _load_lines(TXT_PATH)  # list of {"text": "..."}
    metas = _load_lines(META_PATH) # list of {...}
    # Sanity: align lengths
    if len(texts) != len(metas) or (embs.shape[0] != len(texts)):
        LOG.warning("index alignment mismatch: embs=%s texts=%s metas=%s", embs.shape, len(texts), len(metas))
        n = min(embs.shape[0], len(texts), len(metas))
        embs = embs[:n]
        texts = texts[:n]
        metas = metas[:n]
    return embs, texts, metas

def _cosine_search(query_emb: np.ndarray, embs: np.ndarray, top_k: int = 8, mask: Optional[np.ndarray] = None):
    if embs.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    sims = embs @ query_emb  # embs and query_emb are L2-normalized
    if mask is not None:
        sims = np.where(mask, sims, -1e9)  # mask out
    idx = np.argpartition(-sims, kth=min(top_k, len(sims)-1))[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def search(query: str, k: int = 8, scope: str = "all", meeting_id: str = "") -> List[Tuple[Document, float]]:
    """
    Returns a list of (Document, score) for the query.
    scope: "all" (default) or "this" -> when "this", meeting_id must be provided.
    """
    embs, texts, metas = _load_index()
    if embs.shape[0] == 0:
        LOG.info("search: empty index")
        return []

    # build mask by scope
    mask = None
    if scope == "this" and meeting_id:
        mask = np.array([ (m.get("meeting_id","") == meeting_id) for m in metas ], dtype=bool)

    # embed query
    q_vec = _embed_texts([query], batch_size=1)[0]  # (D,)
    idx, scores = _cosine_search(q_vec, embs, top_k=k, mask=mask)

    out = []
    for i, s in zip(idx, scores):
        doc = Document(page_content=texts[i]["text"], metadata=metas[i])
        out.append((doc, float(s)))
    return out

def keyword_rerank(query: str, hits: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    Simple lexical boost: overlap of query tokens with doc text.
    NewScore = 0.85 * dense + 0.15 * lexical (scaled)
    """
    if not hits:
        return hits
    q_tokens = set([t for t in query.lower().split() if len(t) > 2])
    boosted = []
    for doc, s in hits:
        text = (doc.page_content or "").lower()
        overlap = sum(1 for t in q_tokens if t in text)
        # scale lexical score
        lex = min(overlap / (len(q_tokens) + 1e-6), 1.0)
        new_s = 0.85 * s + 0.15 * lex
        boosted.append((doc, new_s))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

# summarizer/generate_minutes.py
import os
import re
import sys
import json
import time
import math
import argparse
import logging
from typing import List, Tuple, Optional, Dict, Any
# at the top
from string import Template

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------------
# Config (aligned with your train_lora.py)
# ----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
LORA_DIR   = os.getenv("LORA_DIR", "artifacts/flan_t5_minutes_lora")

LOG_DIR_APP = "logs/app"
DEFAULT_CONTEXT_LEN = 512          # encoder max window (base model)
ENC_MAX_TOK = 480                  # a bit under 512 to leave room for special tokens
ENC_OVERLAP = 120                  # token overlap between windows

# Generation budgets
CHUNK_JSON_MAX_NEW  = 360          # per-chunk JSON
FINAL_RENDER_MIN_NEW = 220
FINAL_RENDER_MAX_NEW = 700

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ----------------------------
# Prompts (exact structure you requested)
# ----------------------------
# Map step → STRICT JSON extractor
# replace your current CHUNK_MAP_PROMPT with this:
from string import Template

CHUNK_MAP_PROMPT_TPL = Template("""Return ONLY a VALID JSON ARRAY (starts with `[` and ends with `]`).
Each element MUST be an object with EXACTLY these keys:
- "issue_name" (string)
- "issue" (string)
- "discussion" (array of 3–6 short strings)
- "resort" (string)

Do NOT include any extra text before or after the JSON.
Do NOT include angle brackets, quotes around keys are required.
Do NOT include timestamps or speaker names.
Max 5 discussions per chunk.

Example of VALID SHAPE (content is illustrative only):
[
  {
    "issue_name": "SNCC cultural rooms furniture",
    "issue": "Rooms ready but lack furniture; budget source unclear.",
    "discussion": ["Funding outside Gymkhana.", "JICA window closes 31 March.", "Prepare furniture list with costs."],
    "resort": "List by 28 Jan; review with FiC SA and Dean Planning."
  }
]

Chunk:
<<<
$CHUNK
>>>
""")


# Reduce step → pretty render into repeated 4-section blocks
FINAL_RENDER_PROMPT = """You are formatting meeting minutes.

Given the merged JSON list of discussions, output ONLY repeated blocks in EXACTLY this format:

Issue name: <short title>
Issue: <1–2 lines stating the problem>
Discussion:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Resort: <final decision or next steps>

Rules:
- No timestamps, no speaker names.
- Keep 'Discussion' to 3–6 bullets.
- Do not invent content.
- Leave a blank line between blocks.

JSON:
<<<
{JSON_PAYLOAD}
>>>
"""

# ----------------------------
# Logging
# ----------------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(LORA_DIR) or ".", exist_ok=True)
    os.makedirs("storage", exist_ok=True)
    os.makedirs(LOG_DIR_APP, exist_ok=True)

def setup_logging():
    ensure_dirs()
    log_path = os.path.join(LOG_DIR_APP, "generate_minutes.log")
    # Reset handlers to avoid duplication
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    fh.setLevel(logging.INFO); ch.setLevel(logging.INFO)
    root.setLevel(logging.INFO)
    root.addHandler(fh); root.addHandler(ch)
    logging.info("Logging to %s", log_path)
    return log_path

# ----------------------------
# ASR (optional)
# ----------------------------
def transcribe_audio(audio_path: str) -> str:
    """
    Use faster-whisper if installed; else raise a friendly error.
    Returns a single string transcript (keep timestamps if model outputs them).
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        raise RuntimeError(
            "Audio transcription requires 'faster-whisper'. Install via:\n"
            "  pip install faster-whisper\n"
            "Or provide --transcript instead of --audio."
        )

    model = WhisperModel("small", device=DEVICE if DEVICE != "mps" else "cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, vad_filter=True)
    lines = []
    for seg in segments:
        # Format like: [HH:MM:SS] text
        start = time.strftime("%H:%M:%S", time.gmtime(seg.start))
        lines.append(f"[{start}] {seg.text.strip()}")
    return "\n".join(lines)

# ----------------------------
# Transcript helpers
# ----------------------------
TIMESTAMP_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")

def read_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_transcript(text: str) -> str:
    """
    Light cleanup: normalize whitespace, collapse repeated lines, remove extraneous 'Minutes:' echoes.
    Keep speaker labels and timestamps if present; we'll instruct model not to include them in final minutes.
    """
    text = re.sub(r"(?:\bMinutes:\s*){2,}", "Minutes: ", text, flags=re.IGNORECASE)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # De-dup consecutive identical lines
    lines = []
    prev = None
    for line in text.splitlines():
        if line != prev:
            lines.append(line)
        prev = line
    return "\n".join(lines).strip()

# ----------------------------
# Token chunking
# ----------------------------
def tokenize(tokenizer: AutoTokenizer, text: str) -> List[int]:
    enc = tokenizer(text, truncation=False, add_special_tokens=True)
    return enc["input_ids"]

def detokenize(tokenizer: AutoTokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

def make_windows(ids: List[int], window: int, overlap: int) -> List[Tuple[int, int]]:
    step = max(1, window - overlap)
    spans = []
    for start in range(0, len(ids), step):
        end = min(start + window, len(ids))
        spans.append((start, end))
        if end == len(ids):
            break
    return spans

# ----------------------------
# Model loader (base + LoRA)
# ----------------------------
def load_model_and_tokenizer() -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    # Prefer tokenizer from LORA_DIR (you saved it there), fallback to base
    try:
        tok = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.to(DEVICE)
    model.eval()
    logging.info("Model loaded | base=%s | adapter=%s | device=%s", MODEL_NAME, LORA_DIR, DEVICE)
    return model, tok

# ----------------------------
# Generation helpers
# ----------------------------
DECODE = dict(
    num_beams=6,
    no_repeat_ngram_size=3,
    length_penalty=1.05,
    early_stopping=True,
)

@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens, min_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ENC_MAX_TOK)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=False,                  # <- force deterministic beams
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **DECODE,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# ---- JSON extraction + merge + render helpers ----

_JSON_ARRAY_RE = re.compile(r"\[\s*{.*}\s*\]", re.S)

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract first JSON array from text; return [] on failure."""
    # If the model ever leaks extra tokens, try to slice the first [...].
    m = _JSON_ARRAY_RE.search(text.strip())
    candidate = m.group(0) if m else text.strip()
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            # normalize items
            out = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                out.append({
                    "issue_name": (it.get("issue_name") or "").strip(),
                    "issue": (it.get("issue") or "").strip(),
                    "discussion": [str(b).strip("•- ").strip() for b in (it.get("discussion") or [])][:6],
                    "resort": (it.get("resort") or "").strip() or "Pending — owner + date"
                })
            return out
    except Exception:
        pass
    return []

def summarize_chunks_to_json(model, tokenizer, transcript_text: str) -> List[Dict[str, Any]]:
    ids = tokenize(tokenizer, transcript_text)
    spans = make_windows(ids, window=ENC_MAX_TOK, overlap=ENC_OVERLAP)
    logging.info("Chunking: tokens=%d | windows=%d (win=%d, overlap=%d)", len(ids), len(spans), ENC_MAX_TOK, ENC_OVERLAP)

    all_items: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(spans, 1):
        chunk_txt = detokenize(tokenizer, ids[s:e])
        # in summarize_chunks_to_json(), replace:
        # prompt = CHUNK_MAP_PROMPT.format(CHUNK=chunk_txt)
        prompt = CHUNK_MAP_PROMPT_TPL.substitute(CHUNK=chunk_txt)

        logging.info("Chunk %d/%d | extracting JSON ...", i, len(spans))
        raw = generate_text(model, tokenizer, prompt, max_new_tokens=CHUNK_JSON_MAX_NEW, min_new_tokens=120)

        items = extract_json_array(raw)
        logging.info("Chunk %d | extracted %d discussion(s)", i, len(items))
        all_items.extend(items)
    return all_items

def merge_discussions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate discussions by (issue_name lower, first 16 chars of issue).
    Ensure 3–6 bullets in discussion where possible.
    """
    merged: List[Dict[str, Any]] = []
    seen = set()
    for d in items:
        key = (d.get("issue_name","").strip().lower(), d.get("issue","")[:16].lower())
        if key in seen:
            continue
        seen.add(key)
        disc = d.get("discussion") or []
        # Clamp to 3–6 bullets; if too few, keep as-is (don't invent)
        disc = [b for b in disc if b][:6]
        merged.append({
            "issue_name": d.get("issue_name","").strip(),
            "issue": d.get("issue","").strip(),
            "discussion": disc,
            "resort": d.get("resort","").strip() or "Pending — owner + date"
        })
    return merged

def deterministic_render_blocks(merged: List[Dict[str, Any]]) -> str:
    """Fallback renderer that NEVER fails."""
    lines = []
    for d in merged:
        lines.append(f"Issue name: {d.get('issue_name') or 'NA'}")
        lines.append(f"Issue: {d.get('issue') or 'NA'}")
        lines.append("Discussion:")
        disc = d.get("discussion") or []
        if not disc:
            disc = ["General discussion."]
        for b in disc[:6]:
            lines.append(f"- {b}")
        lines.append(f"Resort: {d.get('resort') or 'Pending — owner + date'}")
        lines.append("")  # blank line between blocks
    return "\n".join(lines).strip()

def render_minutes_blocks(model, tokenizer, merged_json):
    # Deterministic render only (no model) to prevent hallucinations.
    return deterministic_render_blocks(merged_json)


# ----------------------------
# Date helper (optional)
# ----------------------------
DATE_RE_1 = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
def guess_date_from_text(text: str) -> Optional[str]:
    """
    Very light attempt: look for dd/mm/yyyy or dd-mm-yyyy patterns in transcript.
    Returns normalized dd/mm/yyyy if found.
    """
    m = DATE_RE_1.search(text)
    if not m:
        return None
    d, mth, y = m.groups()
    if len(y) == 2:
        y = "20" + y
    d = d.zfill(2); mth = mth.zfill(2)
    return f"{d}/{mth}/{y}"

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    _log_file = setup_logging()

    parser = argparse.ArgumentParser(description="Generate structured 'Issue/Discussion/Resort' minutes from audio or transcript using LoRA adapter.")
    parser.add_argument("--audio", type=str, default="", help="Path to audio file (optional). Requires faster-whisper.")
    parser.add_argument("--transcript", type=str, default="", help="Path to transcript text/markdown file.")
    parser.add_argument("--out", type=str, required=True, help="Output path for final minutes markdown.")
    parser.add_argument("--date", type=str, default="", help="Meeting date in dd/mm/yyyy (optional).")
    parser.add_argument("--scope", type=str, default="this", help="Reserved flag for app integration (this/all).")
    args = parser.parse_args()

    t0 = time.time()
    logging.info("=== generate_minutes start ===")
    logging.info("Args: %s", vars(args))

    if not args.audio and not args.transcript:
        raise SystemExit("Provide either --audio or --transcript.")

    # 1) Obtain transcript text
    if args.transcript:
        raw_text = read_transcript(args.transcript)
        logging.info("Loaded transcript: %s (%d chars)", args.transcript, len(raw_text))
    else:
        raw_text = transcribe_audio(args.audio)
        logging.info("Transcribed audio: %s (%d chars)", args.audio, len(raw_text))

    cleaned = clean_transcript(raw_text)
    # Guess date if not provided (optional; not used in final rendering here)
    meeting_date = args.date.strip() or guess_date_from_text(cleaned) or ""

    # 2) Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()
    # Let tokenizer allow long texts for pre-chunking
    try:
        tokenizer.model_max_length = 10**9
    except Exception:
        pass

    # 3) Map (chunk → JSON arrays of discussions)
    all_items = summarize_chunks_to_json(model, tokenizer, cleaned)

    # Heuristic fallback if model returned nothing (never output raw transcript)
    if not all_items:
        logging.info("No JSON extracted; applying heuristic fallback.")
        # ultra-simple heuristic: treat entire transcript as one discussion
        one = {
            "issue_name": (cleaned.splitlines()[0] if cleaned else "Discussion").strip()[:70],
            "issue": cleaned.replace("\n", " ")[:180],
            "discussion": ["General discussion."],
            "resort": "Pending — owner + date"
        }
        all_items = [one]

    # 4) Reduce (merge/dedupe)
    merged = merge_discussions(all_items)
    if not merged:
        merged = [{
            "issue_name": (cleaned.splitlines()[0] if cleaned else "Discussion")[:70],
            "issue": cleaned.replace("\n", " ")[:180],
            "discussion": ["General discussion."],
            "resort": "Pending — owner + date",
        }]

    # 5) Render (final formatted blocks)
    minutes = render_minutes_blocks(model, tokenizer, merged)

    # 6) Save output
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(minutes.strip() + "\n")
    logging.info("Saved minutes to: %s", out_path)

    # 7) Persist run metadata
    meta = {
        "base_model": MODEL_NAME,
        "adapter_dir": LORA_DIR,
        "device": DEVICE,
        "input_audio": args.audio,
        "input_transcript": args.transcript,
        "out_path": out_path,
        "meeting_date": meeting_date or "<unknown>",
        "tokens_window": ENC_MAX_TOK,
        "tokens_overlap": ENC_OVERLAP,
        "num_discussions": len(merged),
        "elapsed_sec": round(time.time() - t0, 2),
        "log_file": os.path.join(LOG_DIR_APP, "generate_minutes.log")
    }
    meta_path = re.sub(r"\.md$", ".run.json", out_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved run meta: %s", meta_path)
    logging.info("=== generate_minutes done in %.2fs ===", meta["elapsed_sec"])
# ============================================================
# Back-compat wrappers for app.py (do NOT remove)
# ============================================================

_LAST_ITEMS = []  # cache of discussions from the last summarize call

def _chunkwise_json_and_notes(model, tokenizer, transcript_text):
    """Run JSON extractor and also build human-readable notes for the UI."""
    global _LAST_ITEMS
    items = summarize_chunks_to_json(model, tokenizer, transcript_text)
    _LAST_ITEMS = items[:]  # cache for synthesize_minutes()

    # build simple notes list for the debug pane
    notes = []
    for i, it in enumerate(items, 1):
        issue = (it.get("issue_name") or it.get("issue") or "Issue").strip()
        bullets = it.get("discussion") or []
        resort = it.get("resort") or "Pending — owner + date"
        note = (
            f"- Discussion {i}: {issue}\n"
            + "\n".join(f"  • {b}" for b in bullets[:4])
            + f"\n  → Resort: {resort}"
        )
        notes.append(note)
    if not notes:
        notes = ["- No structured items extracted; fallback will render one generic block."]
    return items, notes

def summarize_chunks(model, tokenizer, transcript_text: str):
    """OLD API expected by app.py — returns list[str] of notes."""
    _, notes = _chunkwise_json_and_notes(model, tokenizer, transcript_text)
    return notes

def synthesize_minutes(model, tokenizer, section_summaries, meeting_date=None):
    """OLD API expected by app.py — returns final minutes text."""
    global _LAST_ITEMS
    items = _LAST_ITEMS or []
    if not items:
        joined = "\n".join(section_summaries or [])
        items = [{
            "issue_name": (joined.splitlines()[0] if joined else "Discussion")[:70],
            "issue": joined.replace("\n", " ")[:180],
            "discussion": ["General discussion."],
            "resort": "Pending — owner + date",
        }]
    merged = merge_discussions(items)
    return render_minutes_blocks(model, tokenizer, merged)


if __name__ == "__main__":
    main()

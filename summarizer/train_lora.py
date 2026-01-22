# summarizer/train_lora.py
import os
import re
import json
import logging
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.utils.logging import set_verbosity_info
from peft import LoraConfig, get_peft_model

# -------- Config import with safe fallbacks --------

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
LORA_DIR   = os.getenv("LORA_DIR", "artifacts/flan_t5_minutes_lora")

# ----------------------------
# Basic settings (no TensorBoard)
# ----------------------------
SEED = 42
CONTEXT_LEN = 512      # encoder window
TARGET_LEN  = 256      # minutes cap
TRAIN_JSONL = "data/lora_minutes_train.jsonl"
VAL_JSONL   = "data/lora_minutes_val.jsonl"
LOG_DIR     = "logs/training"
TXT_LOG_FILE = os.path.join(LOG_DIR, "train.txt")  # all logs go here

# Long transcript handling
CHUNK_LONG_DIALOGUES = True
CHUNK_OVERLAP = 128  # tokens overlapped between chunks

# Gradient checkpointing (OFF by default on Mac/MPS)
USE_GRADIENT_CHECKPOINTING = bool(int(os.getenv("USE_GC", "0")))

# Prompt prefix
PROMPT_PREFIX = (
    "You are the official secretary producing IIT Hyderabad Gymkhana Minutes of Meeting.\n"
    "Your role is to generate structured, concise, and professional MoM from raw transcripts.\n"
    "CRITICAL: Do NOT include timestamps of any kind (e.g., 00:00, [00:00]) or 'Chunk' markers.\n\n"
    "Follow this EXACT format:\n\n"
    "MINUTES OF MEETING\n\n"
    "Date: <dd/mm/yyyy>\n"
    "Agenda and Outcomes:\n"
    "1. <Topic>\n"
    "   • Issue: <short description>\n"
    "   • Discussion: <summary of points>\n"
    "   • Resolution/Decision: <final outcome>\n"
    "   • Action Item (if any): <task + responsible person>\n\n"
    "2. <Next Topic>\n"
    "   • Issue: <...>\n"
    "   • Discussion: <...>\n"
    "   • Resolution/Decision: <...>\n"
    "   • Action Item: <...>\n\n"
    "---\n\n"
    "Conversation:\n"
)

# ----------------------------
# Cleaning utilities
# ----------------------------
# [00:00] or 00:00 anywhere; also line-leading versions
TS_ANY = re.compile(r'(?:\[\s*\d{1,2}:\d{2}\s*\]|\b\d{1,2}:\d{2}\b)')
TS_LINE = re.compile(r'(?m)^\s*(?:\[\s*\d{1,2}:\d{2}\s*\]|\d{1,2}:\d{2})\s*')
CHUNK_MARK = re.compile(r'(?im)\bchunk\s*\d+\b[:\-\]]?\s*')
FILLERS = re.compile(r'\b(?:uh+|um+|erm+|you know|like)\b', re.IGNORECASE)

def clean_transcript(text: str) -> str:
    # remove line-leading timestamps
    text = TS_LINE.sub("", text)
    # remove any remaining inline timestamps
    text = TS_ANY.sub("", text)
    # remove "Chunk 1/2/…" markers
    text = CHUNK_MARK.sub("", text)
    # light disfluency cleanup
    text = FILLERS.sub("", text)
    # collapse spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    parent = os.path.dirname(LORA_DIR)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("storage", exist_ok=True)

def setup_logging() -> str:
    """Configure Python logging to write everything into logs/training/train.txt."""
    ensure_dirs()
    # Reset root logger handlers (avoid duplicate logs when re-running)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler
    fh = logging.FileHandler(TXT_LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)

    # Console handler (mirrors file)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)

    # Capture warnings into logging
    logging.captureWarnings(True)
    # HF transformers verbosity
    set_verbosity_info()

    logging.info("Logging initialized. File: %s", TXT_LOG_FILE)
    return TXT_LOG_FILE

def jsonl_load(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def normalize_dialogue(d):
    return "\n".join(d) if isinstance(d, list) else str(d)

# HF Trainer -> logging callback (writes into Python logging)
class TextLoggerCallback(TrainerCallback):
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self._log = logging.getLogger("trainer")

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._log.info(
            "train_begin | output_dir=%s | lr=%.6f | train_bs=%s | eval_bs=%s | epochs=%s | seed=%s",
            args.output_dir, args.learning_rate, args.per_device_train_batch_size,
            args.per_device_eval_batch_size, args.num_train_epochs, args.seed
        )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and (state.global_step % max(1, self.log_every_n_steps) == 0 or "loss" in logs):
            self._log.info("log | step=%s | epoch=%.3f | %s", state.global_step, state.epoch, json.dumps(logs))

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        self._log.info("eval | step=%s | metrics=%s", state.global_step, json.dumps(metrics))

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._log.info("train_end | best_metric=%s", state.best_metric)

# ----------------------------
# Main
# ----------------------------
def main():
    log_path = setup_logging()
    logger = logging.getLogger("train_lora")

    set_seed(SEED)

    # ---------- Load data ----------
    train_items = jsonl_load(TRAIN_JSONL)
    val_items   = jsonl_load(VAL_JSONL)
    logger.info("data loaded | train=%d | val=%d", len(train_items), len(val_items))

    train_ds = Dataset.from_list(train_items)
    val_ds   = Dataset.from_list(val_items)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Prevent full-tokenize warnings during pre-chunking:
    tokenizer.model_max_length = 10**9
    eos_token = tokenizer.eos_token or "</s>"
    logger.info("tokenizer ready | model=%s | eos_token=%r", MODEL_NAME, eos_token)

    # ---------- Preprocess w/ chunking ----------
    def preprocess(batch):
        input_ids_all, attn_all, labels_all = [], [], []

        # Tokenize targets once (targets must NOT contain timestamps)
        target_texts = [str(t).strip() + eos_token for t in batch["summary"]]
        target_enc = tokenizer(
            text_target=target_texts,
            max_length=TARGET_LEN,
            truncation=True,
            padding=False,
        )

        for idx, dia in enumerate(batch["dialogue"]):
            # Clean the dialogue before prefixing (remove timestamps/chunk markers/fillers)
            raw_src = normalize_dialogue(dia)
            cleaned_src = clean_transcript(raw_src)
            src_text = PROMPT_PREFIX + cleaned_src

            # Tokenize full then slice
            full_enc = tokenizer(src_text, truncation=False, add_special_tokens=True)
            full_ids = full_enc["input_ids"]

            if CHUNK_LONG_DIALOGUES and len(full_ids) > CONTEXT_LEN:
                step = max(1, CONTEXT_LEN - CHUNK_OVERLAP)
                for start in range(0, len(full_ids), step):
                    chunk = full_ids[start : start + CONTEXT_LEN]
                    input_ids_all.append(chunk)
                    attn_all.append([1] * len(chunk))
                    labels_all.append(target_enc["input_ids"][idx])
            else:
                short_enc = tokenizer(src_text, max_length=CONTEXT_LEN, truncation=True)
                input_ids_all.append(short_enc["input_ids"])
                attn_all.append(short_enc["attention_mask"])
                labels_all.append(target_enc["input_ids"][idx])

        return {"input_ids": input_ids_all, "attention_mask": attn_all, "labels": labels_all}

    logger.info("tokenizing & chunking | split=train")
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    logger.info("tokenizing & chunking | split=val")
    val_tok   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
    logger.info("expanded samples | train=%d | val=%d", len(train_tok), len(val_tok))

    # ---------- Base model + LoRA ----------
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base.config.use_cache = False  # required for training / checkpointing

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o"],  # T5 attention projections
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base, lora_cfg)

    # Avoid gradient checkpointing on MPS/CPU (can cause no-grad loss on some builds)
    is_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    is_cuda = torch.cuda.is_available()
    if USE_GRADIENT_CHECKPOINTING and is_cuda:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            logger.info("gradient checkpointing ENABLED (CUDA)")
        except Exception as e:
            logger.warning("could not enable gradient checkpointing: %s", e)
    else:
        logger.info("gradient checkpointing DISABLED (default / MPS-safe)")

    # Sanity: show trainables
    try:
        trainable, total = 0, 0
        for p in model.parameters():
            t = p.numel()
            total += t
            if p.requires_grad:
                trainable += t
        logger.info(
            "trainable params: %s || all params: %s || trainable%%: %.4f",
            f"{trainable:,}", f"{total:,}", (trainable / max(1, total)) * 100.0
        )
    except Exception:
        pass

    # ---------- Training args ----------
    args = TrainingArguments(
        output_dir=LORA_DIR,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=4,                 # yes, using epochs
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],                       # no TB/W&B
        logging_dir=LOG_DIR,
        seed=SEED,
        fp16=False,                         # CPU/MPS-safe
        bf16=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=2,      # effective batch size 4
        remove_unused_columns=False,        # safer with custom dicts
        lr_scheduler_type="linear",
        label_smoothing_factor=0.10,        # reduces copy-from-ASR tendency
    )

    # ---------- Data collator ----------
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---------- Persist run config ----------
    run_cfg_path = os.path.join(LOG_DIR, "run_config.json")
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "lora_dir": LORA_DIR,
            "context_len": CONTEXT_LEN,
            "target_len": TARGET_LEN,
            "seed": SEED,
            "train_size": len(train_items),
            "val_size": len(val_items),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "chunk_long_dialogues": CHUNK_LONG_DIALOGUES,
            "chunk_overlap": CHUNK_OVERLAP,
            "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING and is_cuda,
            "device": "cuda" if is_cuda else ("mps" if is_mps else "cpu"),
        }, f, indent=2)
    logger.info("run_config saved | %s", run_cfg_path)

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok if len(val_tok) > 0 else None,
        tokenizer=tokenizer,  # might warn deprecation in future; fine on 4.56.x
        data_collator=collator,
        callbacks=[TextLoggerCallback(log_every_n_steps=args.logging_steps)],
    )

    # ---------- Train ----------
    try:
        trainer.train()
    except Exception as e:
        logging.getLogger("trainer").exception("exception during trainer.train: %s", e)
        raise

    # ---------- Save ----------
    model.save_pretrained(LORA_DIR)   # adapter_config.json + adapter_model.bin
    tokenizer.save_pretrained(LORA_DIR)
    logger.info("saved | lora_dir=%s", LORA_DIR)
    logger.info("log_file | %s", log_path)

if __name__ == "__main__":
    main()


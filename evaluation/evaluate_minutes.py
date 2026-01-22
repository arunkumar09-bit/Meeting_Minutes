# evaluation/evaluate_minutes.py
import json, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer

def has_structure(text: str) -> bool:
    MUST = [
        r"MINUTES OF MEETING",
        r"Date:\s*",
        r"Agenda and outcomes:",
        r"\n1\.\s",               # at least one numbered item
        r"•\s*Issue:",
        r"•\s*Discussion:",
        r"(•\s*Resort:|•\s*Resolution/Decision:)"
    ]
    return all(re.search(p, text, re.I) for p in MUST)

def load_model(model_name, lora_dir):
    tok = AutoTokenizer.from_pretrained(lora_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, lora_dir)
    return tok, model

def evaluate(jsonl_path, model_name, lora_dir, n=20):
    data = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
    ds = Dataset.from_list(data[:n])
    tok, model = load_model(model_name, lora_dir)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores, fmt_ok = [], 0
    for ex in ds:
        inp = tok("Conversation:\n" + ex["dialogue"], return_tensors="pt", truncation=True, max_length=1024)
        out = model.generate(**inp, max_new_tokens=600, num_beams=4)
        pred = tok.decode(out[0], skip_special_tokens=True)
        scores.append(scorer.score(ex["summary"], pred)["rougeL"].fmeasure)
        fmt_ok += int(has_structure(pred))

    print(f"ROUGE-L (mean on {len(scores)}): {sum(scores)/len(scores):.3f}")
    print(f"Format compliance: {fmt_ok}/{len(scores)}")

if __name__ == "__main__":
    import os
    from config import MODEL_NAME, LORA_DIR
    evaluate("data/lora_minutes_val.jsonl", MODEL_NAME, LORA_DIR)

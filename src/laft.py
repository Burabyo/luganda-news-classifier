"""Language-Adaptive Fine-Tuning (LAFT) of AfroXLMR-base on unlabeled Luganda text.

AfroXLMR-base never saw Luganda during pretraining. LAFT (Pfeiffer et al., 2020; Alabi et al., 2022)
continues the masked-language-model (MLM) pretraining objective on target-language text so that
the encoder adapts to Luganda vocabulary and morphology *before* it is fine-tuned on the
small labelled set. The labelled data is not used here, and neither is any dev/test text.

Objective: randomly mask 15% of subword tokens (80% [MASK] / 10% random / 10% unchanged) and
train the model to predict the original tokens (cross-entropy over the vocabulary).

Corpus: the same Luganda-filtered CC-100 + Wikipedia text used for Word2Vec (cased, untokenised).
Run: python src/laft.py   -> models/afroxlmr_base_laft_lug/
"""
import json
import math
import random
import time

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from data import MODELS_DIR, RESULTS_DIR, normalize, tokenize
from word2vec_train import UNLAB, is_luganda

BASE = str(MODELS_DIR / "pretrained" / "Davlan__afro-xlmr-base")
OUT = MODELS_DIR / "afroxlmr_base_laft_lug"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LEN, BS, LR, MAX_LINES = 128, 16, 5e-5, 60_000


def corpus():
    lines = [normalize(l) for l in open(UNLAB / "cc100_lg.txt", encoding="utf-8", errors="ignore")]
    wiki = pd.read_parquet(UNLAB / "wiki_lg.parquet")
    lines += [normalize(p) for doc in wiki["text"] for p in doc.split("\n")]
    lines = [l for l in dict.fromkeys(lines) if is_luganda(tokenize(l))]  # dedupe + language filter
    random.Random(0).shuffle(lines)
    return lines[:MAX_LINES]


def main():
    torch.manual_seed(0)
    lines = corpus()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMaskedLM.from_pretrained(BASE).to(DEVICE)
    enc = tok(lines, truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": ids} for ids in enc["input_ids"]]
    held = items[:1000]       # held-out unlabeled lines to measure MLM perplexity before/after
    train = items[1000:]
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    dl = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True, collate_fn=coll)
    hl = torch.utils.data.DataLoader(held, batch_size=32, collate_fn=coll)

    @torch.no_grad()
    def perplexity():
        torch.manual_seed(1)  # same masks each time
        model.eval()
        losses = [model(**{k: v.to(DEVICE) for k, v in b.items()}).loss.item() for b in hl]
        model.train()
        return math.exp(sum(losses) / len(losses))

    stats = {"lines": len(lines), "train_lines": len(train), "ppl_before": perplexity()}
    print(stats, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl)), len(dl))
    t0 = time.time()
    model.train()
    for step, b in enumerate(dl, 1):
        loss = model(**{k: v.to(DEVICE) for k, v in b.items()}).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 200 == 0:
            print(f"step {step}/{len(dl)} loss={loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    stats["ppl_after"] = perplexity()
    stats["steps"] = len(dl)
    stats["seconds"] = time.time() - t0
    stats["config"] = {"base": BASE, "max_len": MAX_LEN, "batch_size": BS, "lr": LR, "epochs": 1, "mlm_prob": 0.15}
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    with open(RESULTS_DIR / "laft_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(stats)


if __name__ == "__main__":
    main()

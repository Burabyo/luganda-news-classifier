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
# Physical batch 8 x 2 accumulation steps = effective batch 16 (memory limit of a 16 GB laptop).
MAX_LEN, BS, ACCUM, LR, MAX_LINES = 128, 8, 2, 5e-5, 25_000


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
    # Memory: the 250k x 768 word-embedding matrix (tied to the MLM output layer) is 192M of the
    # model's 278M parameters. Freezing it removes most optimizer state so LAFT fits on a 16 GB laptop;
    # all 12 Transformer layers (where contextual knowledge lives) are still adapted to Luganda.
    model.get_input_embeddings().weight.requires_grad = False
    stats_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = tok(lines, truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": ids} for ids in enc["input_ids"]]
    held = items[:1000]       # held-out unlabeled lines to measure MLM perplexity before/after
    train = items[1000:]
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    dl = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True, collate_fn=coll)
    hl = torch.utils.data.DataLoader(held, batch_size=32, collate_fn=coll)

    def mlm_loss(b):
        """MLM loss computed only at masked positions (labels != -100), so the 250k-way output layer
        runs on ~15% of tokens instead of all of them. Mathematically identical to the full loss."""
        b = {k: v.to(DEVICE) for k, v in b.items()}
        hidden = model.base_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        sel = b["labels"] != -100
        logits = model.lm_head(hidden[sel])
        return torch.nn.functional.cross_entropy(logits, b["labels"][sel])

    @torch.no_grad()
    def perplexity():  # exp(mean masked-token cross-entropy) on 1,000 held-out lines
        torch.manual_seed(1)  # same masks each time
        model.eval()
        losses = [mlm_loss(b).item() for b in hl]
        model.train()
        return math.exp(sum(losses) / len(losses))

    stats = {"lines": len(lines), "train_lines": len(train), "trainable_params": stats_trainable,
             "ppl_before": perplexity()}
    print(stats, flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    n_updates = len(dl) // ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_updates), n_updates)
    t0 = time.time()
    model.train()
    for step, b in enumerate(dl, 1):
        loss = mlm_loss(b)
        (loss / ACCUM).backward()
        if step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if step % 800 == 0 or step == 40:
            print(f"step {step}/{len(dl)} loss={loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    stats["ppl_after"] = perplexity()
    stats["updates"] = n_updates
    stats["seconds"] = time.time() - t0
    stats["config"] = {"base": "Davlan/afro-xlmr-base", "max_len": MAX_LEN, "batch_size": BS * ACCUM, "grad_accumulation": ACCUM, "lr": LR, "epochs": 1, "mlm_prob": 0.15}
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    with open(RESULTS_DIR / "laft_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(stats)


if __name__ == "__main__":
    main()

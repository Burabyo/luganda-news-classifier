"""Fine-tune a pretrained Transformer encoder for Luganda news topic classification.

Model: <pretrained encoder> + classification head (AutoModelForSequenceClassification):
  text -> SentencePiece subword tokens (max_len) -> 12 (or 8) self-attention layers
       -> representation of the first token <s> -> dense + tanh -> dropout -> Linear(hidden, 5)
Training: AdamW, linear warm-up (10%) then linear decay, cross-entropy, gradient clipping 1.0,
model selection on dev macro-F1 after every epoch (the test set is only scored with the best epoch).

Options used by the experiments:
  --aux run,swa     add Kirundi/Swahili MasakhaNEWS training articles (cross-lingual transfer)
  --field headline  train/test on headlines only (input ablation)
  --lora            parameter-efficient fine-tuning with LoRA (Hu et al., 2022) instead of full FT
  --save DIR        keep the best checkpoint (used for the deployed model)

Run example: python src/transformer_ft.py --model Davlan/afro-xlmr-base --name afroxlmr_base
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import ID2LABEL, LABEL2ID, MODELS_DIR, load_auxiliary, load_lug
from metrics import compute, full_report, save_run

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(tok, df, max_len, bs, shuffle, seed=0):
    enc = tok(list(df["input"]), truncation=True, max_length=max_len)
    labels = df["category"].map(LABEL2ID).values
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])}
             for i in range(len(df))]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch], return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items() if k != "labels"}
        probs.append(torch.softmax(model(**b).logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def run(args, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train, dev, test = load_lug(args.field)
    if args.aux:
        aux = load_auxiliary(args.aux.split(","), args.field)
        train = pd.concat([train, aux], ignore_index=True)
    y_dev = dev["category"].map(LABEL2ID).values
    y_te = test["category"].map(LABEL2ID).values

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.1,
                         target_modules=["query", "value"])
        model = get_peft_model(model, cfg)  # the classifier head stays fully trainable
    model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    tr_dl = make_loader(tok, train, args.max_len, args.bs, True, seed)
    dev_dl = make_loader(tok, dev, args.max_len, 32, False)
    te_dl = make_loader(tok, test, args.max_len, 32, False)
    # subword fertility: how many pieces the tokenizer needs per Luganda word (higher = worse fit)
    sample = test["input"].head(100)
    fertility = np.mean([len(tok.tokenize(t)) / max(1, len(t.split())) for t in sample])
    truncated = np.mean([len(x) >= args.max_len for x in tok(list(test["input"]))["input_ids"]])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr_dl.dataset) / args.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, best_state, history, bad = -1, None, [], 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for b in tr_dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * len(b["labels"]); n += len(b["labels"])
        dev_f1 = compute(y_dev, predict(model, dev_dl).argmax(1))["macro_f1"]
        history.append({"epoch": ep + 1, "train_loss": tot / n, "dev_macro_f1": dev_f1})
        print(f"  [{args.name} s{seed}] epoch {ep + 1} loss={tot / n:.4f} dev_f1={dev_f1:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if dev_f1 > best:
            best, bad = dev_f1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    probs = predict(model, te_dl)
    pred = probs.argmax(1)
    test_m = compute(y_te, pred)
    rec = {"model": args.name, "family": "transformer", "seed": seed, "pretrained": args.model,
           "config": {k: v for k, v in vars(args).items() if k not in ("seeds", "save")},
           "train_size": len(train), "trainable_params": n_train, "total_params": n_total,
           "subword_fertility": float(fertility), "test_truncated_share": float(truncated),
           "dev_macro_f1": best, "history": history, "test": test_m, **full_report(y_te, pred),
           "train_seconds": time.time() - t0}
    save_run(f"{args.name}_s{seed}", rec, preds=pred, probs=probs)
    print(f"{args.name} seed={seed} dev={best:.4f} test F1={test_m['macro_f1']:.4f} "
          f"acc={test_m['accuracy']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if args.save:
        out = MODELS_DIR / args.save
        if args.lora:
            model = model.merge_and_unload()
        model.save_pretrained(out); tok.save_pretrained(out)
    del model, opt
    if DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="13,42,123")
    ap.add_argument("--field", default="headline+text")
    ap.add_argument("--aux", default="")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    for s in map(int, args.seeds.split(",")):
        run(args, s)


if __name__ == "__main__":
    main()

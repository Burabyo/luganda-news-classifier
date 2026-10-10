"""Recurrent classifiers: BiLSTM / BiGRU with additive attention pooling.

Architecture (input -> output):
  token ids [B, T]
   -> Embedding [B, T, 300]   (random init, or initialised from our Luganda Word2Vec)
   -> 1-layer bidirectional LSTM/GRU, hidden 128 per direction  -> H [B, T, 256]
   -> additive attention (Bahdanau et al., 2015 style): score_t = v^T tanh(W h_t);
      alpha = softmax(score over non-pad tokens); context = sum_t alpha_t h_t   [B, 256]
   -> dropout -> Linear(256, 5) -> logits
Loss: class-weighted cross-entropy (inverse class frequency) to counter the politics majority.

Experiments (each run with 3 seeds, model selected on dev macro-F1 with early stopping):
  bilstm_random     random embeddings, vocabulary = training words (min count 2)
  bilstm_w2v_frozen Word2Vec embeddings, frozen, vocabulary = train words + Word2Vec vocabulary
  bilstm_w2v_tuned  Word2Vec embeddings, fine-tuned
  bigru_w2v_tuned   same as above with a GRU instead of an LSTM

Run: python src/rnn.py [--only NAME]
"""
import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

from data import LABEL2ID, LABELS, MODELS_DIR, load_lug, tokenize
from metrics import compute, full_report, save_run

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 400  # covers the 95th percentile of article length (~360 tokens)
PAD, UNK = 0, 1


class Vocab:
    def __init__(self, words):
        self.itos = ["<pad>", "<unk>"] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:MAX_LEN]
        return ids or [UNK]


class AttnRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden=128, n_classes=5, cell="lstm",
                 emb_weights=None, freeze_emb=False, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if emb_weights is not None:
            self.emb.weight.data.copy_(torch.as_tensor(emb_weights))
        self.emb.weight.requires_grad = not freeze_emb
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn_W = nn.Linear(2 * hidden, 2 * hidden)
        self.attn_v = nn.Linear(2 * hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, return_attn=False):
        mask = x != PAD
        h, _ = self.rnn(self.drop(self.emb(x)))                      # [B, T, 2H]
        scores = self.attn_v(torch.tanh(self.attn_W(h))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)             # [B, 2H]
        logits = self.out(self.drop(ctx))
        return (logits, alpha) if return_attn else logits


def batches(X, y, bs, shuffle, rng):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), bs):
        b = idx[i:i + bs]
        L = max(len(X[j]) for j in b)
        xb = torch.full((len(b), L), PAD, dtype=torch.long)
        for k, j in enumerate(b):
            xb[k, :len(X[j])] = torch.tensor(X[j])
        yield xb.to(DEVICE), torch.tensor(y[b]).to(DEVICE)


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    probs = []
    for xb, _ in batches(X, np.zeros(len(X), dtype=int), bs, False, None):
        probs.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(probs)


def run(name, cell, emb, freeze, seed, data, w2v, epochs=25, patience=5, lr=1e-3, bs=32):
    train, dev, test, y_tr, y_dev, y_te = data
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)

    counts = Counter(t for x in train["input"] for t in tokenize(x))
    words = {w for w, c in counts.items() if c >= 2}
    if emb == "w2v":
        words |= set(w2v.index_to_key)  # unlabeled-corpus vocabulary: reduces test OOV
    vocab = Vocab(words)
    weights = None
    if emb == "w2v":
        weights = np.random.RandomState(seed).normal(0, 0.1, (len(vocab.itos), 300)).astype("float32")
        weights[PAD] = 0
        for w, i in vocab.stoi.items():
            if w in w2v:
                weights[i] = w2v[w]
    Xtr, Xdev, Xte = ([vocab.encode(t) for t in d["input"]] for d in (train, dev, test))
    oov = np.mean([i == UNK for x in Xte for i in x])

    model = AttnRNN(len(vocab.itos), cell=cell, emb_weights=weights, freeze_emb=freeze).to(DEVICE)
    cw = torch.tensor(len(y_tr) / (len(LABELS) * np.bincount(y_tr)), dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    best, best_state, bad, history = -1, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for xb, yb in batches(Xtr, y_tr, bs, True, rng):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
        dev_f1 = compute(y_dev, predict(model, Xdev).argmax(1))["macro_f1"]
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "dev_macro_f1": dev_f1})
        if dev_f1 > best:
            best, bad = dev_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    probs = predict(model, Xte)
    pred = probs.argmax(1)
    test_m = compute(y_te, pred)
    rec = {"model": name, "family": "rnn", "seed": seed,
           "config": {"cell": cell, "embeddings": emb, "frozen": freeze, "hidden": 128, "emb_dim": 300,
                      "lr": lr, "batch_size": bs, "max_len": MAX_LEN, "dropout": 0.5, "patience": patience},
           "vocab_size": len(vocab.itos), "test_oov_rate": float(oov),
           "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "dev_macro_f1": best, "history": history, "test": test_m, **full_report(y_te, pred),
           "train_seconds": time.time() - t0}
    save_run(f"{name}_s{seed}", rec, preds=pred, probs=probs)
    print(f"{name} seed={seed} oov={oov:.3f} dev={best:.4f} test F1={test_m['macro_f1']:.4f} "
          f"acc={test_m['accuracy']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model, vocab


EXPERIMENTS = {
    "bilstm_random": ("lstm", "random", False),
    "bilstm_w2v_frozen": ("lstm", "w2v", True),
    "bilstm_w2v_tuned": ("lstm", "w2v", False),
    "bigru_w2v_tuned": ("gru", "w2v", False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default="13,42,123")
    args = ap.parse_args()
    train, dev, test = load_lug()
    y = [d["category"].map(LABEL2ID).values for d in (train, dev, test)]
    data = (train, dev, test, *y)
    w2v = KeyedVectors.load(str(MODELS_DIR / "w2v_lug_300d.kv"))
    for name, (cell, emb, freeze) in EXPERIMENTS.items():
        if args.only and name != args.only:
            continue
        for s in map(int, args.seeds.split(",")):
            run(name, cell, emb, freeze, s, data, w2v)


if __name__ == "__main__":
    main()

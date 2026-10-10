"""Error analysis for the best model (and a comparison with the best classical baseline).

Produces:
  results/figures/confusion_<model>.png      normalised confusion matrix
  results/figures/per_class_f1.png           per-class F1, baseline vs best model
  results/figures/confidence_vs_correct.png  calibration-style view of errors
  results/error_analysis.json                error buckets, length/OOV effects, agreement stats
  results/misclassified_<model>.csv          every test error with headline, gold, predicted, confidence

Run: python src/error_analysis.py [--model afroxlmr_base_laft_s42] [--baseline tfidf_word+char_svm]
"""
import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import LABEL2ID, LABELS, RESULTS_DIR, load_lug, tokenize

FIG = RESULTS_DIR / "figures"


def best_run(prefix_filter=None):
    """Pick the transformer run with the best *dev* F1 (never choose by test score)."""
    best = None
    for p in (RESULTS_DIR / "runs").glob("*.json"):
        r = json.load(open(p))
        if r["family"] != "transformer" or (prefix_filter and not p.stem.startswith(prefix_filter)):
            continue
        if best is None or r["dev_macro_f1"] > best[1]["dev_macro_f1"]:
            best = (p.stem, r)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--baseline", default="tfidf_word+char_svm")
    args = ap.parse_args()
    name = args.model or best_run()
    train, _, test = load_lug()
    y = test["category"].map(LABEL2ID).values
    pred = np.load(RESULTS_DIR / "predictions" / f"{name}_pred.npy")
    probs = np.load(RESULTS_DIR / "predictions" / f"{name}_probs.npy")
    base = np.load(RESULTS_DIR / "predictions" / f"{args.baseline}_pred.npy")
    conf = probs.max(1)
    out = {"model": name, "baseline": args.baseline, "n_test": len(y),
           "model_errors": int((pred != y).sum()), "baseline_errors": int((base != y).sum())}

    # Agreement between the neural model and the baseline: are they making the same mistakes?
    out["both_wrong"] = int(((pred != y) & (base != y)).sum())
    out["only_model_wrong"] = int(((pred != y) & (base == y)).sum())
    out["only_baseline_wrong"] = int(((pred == y) & (base != y)).sum())

    # Most frequent confusions (gold -> predicted)
    pairs = Counter((LABELS[g], LABELS[p]) for g, p in zip(y, pred) if g != p)
    out["top_confusions"] = [{"gold": g, "pred": p, "count": c} for (g, p), c in pairs.most_common(10)]

    # Does the model fail more on long articles (truncation) or OOV-heavy articles?
    vocab = set(t for x in train["input"] for t in tokenize(x))
    test = test.copy()
    test["n_tokens"] = test["input"].map(lambda t: len(tokenize(t)))
    test["oov_rate"] = test["input"].map(lambda t: np.mean([w not in vocab for w in tokenize(t)]))
    test["correct"] = pred == y
    test["confidence"] = conf
    test["length_bin"] = pd.cut(test["n_tokens"], [0, 150, 250, 350, 2000], labels=["<150", "150-250", "250-350", ">350"])
    out["accuracy_by_length"] = test.groupby("length_bin", observed=False)["correct"].agg(["mean", "count"]).round(3).to_dict()
    test["oov_bin"] = pd.qcut(test["oov_rate"], 3, labels=["low OOV", "mid OOV", "high OOV"])
    out["accuracy_by_oov_tercile"] = test.groupby("oov_bin", observed=False)["correct"].agg(["mean", "count"]).round(3).to_dict()
    out["mean_conf_correct"] = float(conf[pred == y].mean())
    out["mean_conf_wrong"] = float(conf[pred != y].mean())
    out["errors_with_conf_below_0.6"] = int(((pred != y) & (conf < 0.6)).sum())

    # Save every error for manual inspection (the report discusses these by hand)
    err = test[~test["correct"]].assign(gold=[LABELS[i] for i in y[pred != y]],
                                        predicted=[LABELS[i] for i in pred[pred != y]],
                                        baseline_pred=[LABELS[i] for i in base[pred != y]])
    err[["headline", "gold", "predicted", "baseline_pred", "confidence", "n_tokens", "oov_rate", "url", "text"]] \
        .sort_values("confidence", ascending=False).to_csv(RESULTS_DIR / f"misclassified_{name}.csv", index=False)
    # Confident correct examples, for the "what works" discussion
    ok = test[test["correct"]].assign(gold=[LABELS[i] for i in y[pred == y]]).sort_values("confidence", ascending=False)
    out["confident_correct_examples"] = ok.groupby("gold").head(1)[["gold", "headline", "confidence"]].to_dict("records")

    # Figures
    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(y, pred, normalize="true")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("predicted"); plt.ylabel("gold"); plt.title(f"Confusion matrix (row-normalised)\n{name}")
    plt.tight_layout(); plt.savefig(FIG / f"confusion_{name}.png", dpi=200); plt.close()

    cmb = confusion_matrix(y, base, normalize="true")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cmb, annot=True, fmt=".2f", cmap="Oranges", xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("predicted"); plt.ylabel("gold"); plt.title(f"Confusion matrix (row-normalised)\n{args.baseline}")
    plt.tight_layout(); plt.savefig(FIG / f"confusion_{args.baseline}.png", dpi=200); plt.close()

    f_m = f1_score(y, pred, average=None, labels=range(5))
    f_b = f1_score(y, base, average=None, labels=range(5))
    out["per_class_f1"] = {l: {"model": float(a), "baseline": float(b)} for l, a, b in zip(LABELS, f_m, f_b)}
    x = np.arange(5)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, f_b, 0.4, label=args.baseline, color="#4C72B0")
    plt.bar(x + 0.2, f_m, 0.4, label=name, color="#55A868")
    plt.xticks(x, LABELS); plt.ylim(0, 1.05); plt.ylabel("F1"); plt.legend(loc="lower right")
    plt.title("Per-class F1 on the Luganda test set"); plt.tight_layout()
    plt.savefig(FIG / "per_class_f1.png", dpi=200); plt.close()

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0.2, 1, 17)
    plt.hist(conf[pred == y], bins=bins, alpha=0.7, label="correct", color="#55A868")
    plt.hist(conf[pred != y], bins=bins, alpha=0.8, label="wrong", color="#C44E52")
    plt.xlabel("model confidence (max softmax probability)"); plt.ylabel("test articles"); plt.legend()
    plt.title("Errors concentrate at low confidence"); plt.tight_layout()
    plt.savefig(FIG / "confidence_vs_correct.png", dpi=200); plt.close()

    with open(RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "confident_correct_examples"}, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

"""Collect results/runs/*.json into the experiment table (mean ± std over seeds) and figures.

Outputs: results/experiments.csv, results/experiments.md, results/figures/experiments_f1.png
Run: python src/aggregate_results.py
"""
import json
import re

import matplotlib.pyplot as plt
import pandas as pd

from data import RESULTS_DIR

# Order and descriptions used in the report's experiment table
EXPERIMENTS = [
    ("E0", "majority", "Majority class (always 'politics')"),
    ("E1", "tfidf_word_nb", "TF-IDF words + Complement Naive Bayes"),
    ("E2", "tfidf_word_lr", "TF-IDF words + Logistic Regression (baseline)"),
    ("E3", "tfidf_char_lr", "TF-IDF char 2-4-grams + Logistic Regression"),
    ("E4", "tfidf_word+char_svm", "TF-IDF words + chars + Linear SVM"),
    ("E5", "bilstm_random", "BiLSTM + attention, random embeddings"),
    ("E6", "bilstm_w2v_frozen", "BiLSTM + attention, Word2Vec (frozen)"),
    ("E7", "bilstm_w2v_tuned", "BiLSTM + attention, Word2Vec (fine-tuned)"),
    ("E8", "bigru_w2v_tuned", "BiGRU + attention, Word2Vec (fine-tuned)"),
    ("E9", "xlmr_base", "XLM-R-base (no Luganda in pretraining)"),
    ("E10", "xlmr_base_stable", "XLM-R-base, stabilised (lr 1e-5, 15 epochs)"),
    ("E11", "afriberta_base", "AfriBERTa-base (no Luganda)"),
    ("E12", "afroxlmr_base", "AfroXLMR-base (no Luganda)"),
    ("E13", "serengeti", "Serengeti (Luganda in pretraining)"),
    ("E14", "afroxlmr_base_laft", "AfroXLMR-base + our Luganda LAFT"),
    ("E15", "afroxlmr_base_aux", "AfroXLMR-base + Kirundi/Swahili train data"),
    ("E16", "afroxlmr_base_lora", "AfroXLMR-base, LoRA r=16 (PEFT)"),
    ("E17", "afroxlmr_base_headline", "AfroXLMR-base, headline only"),
]


def load_runs():
    rows = []
    for p in sorted((RESULTS_DIR / "runs").glob("*.json")):
        r = json.load(open(p))
        name = re.sub(r"_s\d+$", "", p.stem)
        rows.append({"name": name, "seed": r.get("seed", 0), "dev_f1": r["dev_macro_f1"],
                     **{f"test_{k}": v for k, v in r["test"].items()},
                     "trainable_params": r.get("trainable_params"), "train_seconds": r.get("train_seconds")})
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    out = []
    for eid, name, desc in EXPERIMENTS:
        g = runs[runs.name == name]
        if g.empty:
            continue
        row = {"id": eid, "name": name, "description": desc, "seeds": len(g)}
        for m in ["dev_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_weighted_f1"]:
            row[m] = g[m].mean()
            row[m + "_std"] = g[m].std(ddof=0) if len(g) > 1 else 0.0
        row["trainable_params"] = g["trainable_params"].iloc[0]
        row["train_minutes"] = g["train_seconds"].mean() / 60
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_DIR / "experiments.csv", index=False)

    def pm(r, m):
        return f"{100 * r[m]:.1f} ± {100 * r[m + '_std']:.1f}" if r["seeds"] > 1 else f"{100 * r[m]:.1f}"
    lines = ["| ID | Approach | Seeds | Dev macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |", "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['id']} | {r['description']} | {r['seeds']} | {pm(r, 'dev_f1')} | "
                     f"{pm(r, 'test_accuracy')} | {pm(r, 'test_macro_f1')} | {pm(r, 'test_weighted_f1')} |")
    (RESULTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fam = lambda i: "#8C8C8C" if i == "E0" else ("#4C72B0" if int(i[1:]) <= 4 else ("#DD8452" if int(i[1:]) <= 8 else "#55A868"))
    fig, ax = plt.subplots(figsize=(9, 6))
    d = df.iloc[::-1]
    ax.barh(d["id"] + "  " + d["description"], 100 * d["test_macro_f1"], xerr=100 * d["test_macro_f1_std"],
            color=[fam(i) for i in d["id"]], capsize=3)
    ax.set_xlabel("Test macro-F1 (%)  — mean ± std over seeds")
    ax.set_xlim(0, 100)
    for y, (v, sd) in enumerate(zip(d["test_macro_f1"], d["test_macro_f1_std"])):
        ax.text(100 * (v + sd) + 1.5, y, f"{100 * v:.1f}", va="center", fontsize=8)
    ax.set_title("Luganda news topic classification: all experiments")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / "experiments_f1.png", dpi=200)


if __name__ == "__main__":
    main()

"""Exploratory data analysis of the Luganda MasakhaNEWS split.

Outputs: results/eda.json and figures in results/figures/.
Run: python src/eda.py
"""
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import LABELS, RAW_DIR, RESULTS_DIR, load_lug, tokenize

FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    train, dev, test = load_lug()
    stats = {}

    # 1. Size and class balance
    dist = pd.DataFrame({s: d["category"].value_counts() for s, d in
                         [("train", train), ("dev", dev), ("test", test)]}).loc[LABELS]
    stats["split_sizes"] = {"train": len(train), "dev": len(dev), "test": len(test)}
    stats["class_distribution"] = dist.to_dict()
    stats["imbalance_ratio_train"] = float(dist["train"].max() / dist["train"].min())

    ax = dist.plot(kind="bar", figsize=(7, 4), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Luganda MasakhaNEWS: articles per category")
    ax.set_ylabel("articles"); ax.set_xlabel("")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=200); plt.close()

    # 2. Length (whitespace tokens)
    for d in (train, dev, test):
        d["n_tokens"] = d["input"].map(lambda t: len(tokenize(t)))
    stats["tokens_per_article"] = train["n_tokens"].describe().round(1).to_dict()
    stats["headline_tokens_mean"] = float(train["headline"].map(lambda t: len(tokenize(t))).mean())
    stats["tokens_by_class_mean"] = train.groupby("category")["n_tokens"].mean().round(1).to_dict()
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=train, x="category", y="n_tokens", order=LABELS)
    plt.title("Article length by category (train)"); plt.ylabel("tokens"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(FIG / "length_by_class.png", dpi=200); plt.close()

    # 3. Vocabulary: Luganda is agglutinative, so the vocabulary is big and sparse
    tr_tokens = [t for x in train["input"] for t in tokenize(x)]
    vocab = Counter(tr_tokens)
    te_tokens = [t for x in test["input"] for t in tokenize(x)]
    oov = sum(1 for t in te_tokens if t not in vocab)
    te_types = set(te_tokens)
    stats["train_tokens"] = len(tr_tokens)
    stats["train_types"] = len(vocab)
    stats["type_token_ratio"] = len(vocab) / len(tr_tokens)
    stats["hapax_share_of_types"] = sum(1 for c in vocab.values() if c == 1) / len(vocab)
    stats["test_oov_token_rate"] = oov / len(te_tokens)
    stats["test_oov_type_rate"] = len([t for t in te_types if t not in vocab]) / len(te_types)

    # 4. Data quality checks
    raw = pd.read_csv(RAW_DIR / "lug" / "train.tsv", sep="\t")
    stats["raw_apostrophe_variants"] = {
        "U+2019 (’)": int(raw["text"].str.count("’").sum()),
        "U+0027 (')": int(raw["text"].str.count("'").sum()),
        "U+2018 (‘)": int(raw["text"].str.count("‘").sum()),
    }
    stats["raw_replacement_chars"] = int(raw["text"].str.count("�").sum())
    stats["source_domains"] = raw["url"].str.extract(r"https?://([^/]+)")[0].value_counts().to_dict()
    all_text = pd.concat([train, dev, test])
    stats["exact_duplicate_texts_across_splits"] = int(all_text["text"].duplicated().sum())
    tr_heads = set(train["headline"])
    stats["test_headlines_seen_in_train"] = int(test["headline"].isin(tr_heads).sum())
    stats["articles_with_english_words"] = int(train["text"].str.contains(
        r"\b(the|and|of|government|president)\b", flags=re.I).sum())

    # 5. Most frequent informative words per class (ignore words common to all classes)
    common = {w for w, _ in vocab.most_common(60)}
    top = {}
    for lab in LABELS:
        c = Counter(t for x in train.loc[train.category == lab, "input"] for t in tokenize(x)
                    if t not in common and len(t) > 3)
        top[lab] = [w for w, _ in c.most_common(12)]
    stats["top_words_per_class"] = top

    with open(RESULTS_DIR / "eda.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

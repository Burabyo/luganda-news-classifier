"""Train Luganda Word2Vec (skip-gram) embeddings on unlabeled text.

Corpus = filtered CC-100 Luganda (Conneau et al., 2020; Wenzek et al., 2020)
       + Luganda Wikipedia (wikimedia/wikipedia, 20231101.lg)
       + Luganda MasakhaNEWS *train* articles (never dev/test).

CC-100 'lg' is very noisy: many lines are English, boilerplate or byte garbage. That matches the
audit of Kreutzer et al. (2022), who found many low-resource web corpora contain little usable
in-language text. We therefore keep a line only if it looks like Luganda (see `is_luganda`).
We also check that no dev/test article leaks into the unlabeled corpus.

Run: python src/word2vec_train.py
"""
import json
import re

import pandas as pd
from gensim.models import Word2Vec

from data import MODELS_DIR, RESULTS_DIR, ROOT, load_lug, tokenize

UNLAB = ROOT / "data" / "unlabeled"

# High-frequency Luganda function words / particles (conjunctions, prepositions, copula, etc.)
LUG_FUNC = {"mu", "ku", "nga", "era", "ne", "naye", "kubanga", "ate", "bwe", "nti", "kye", "kyo",
            "ye", "wa", "ba", "bya", "kya", "gwa", "lwa", "za", "eri", "kati", "nnyo", "buli",
            "awamu", "wano", "okuva", "olw'", "n'", "y'", "ky'", "w'", "ab'", "tebya", "naddala"}
EN_FUNC = {"the", "and", "of", "to", "is", "in", "that", "for", "with", "on", "are", "this", "was"}


def is_luganda(tokens: list[str]) -> bool:
    if len(tokens) < 6:
        return False
    alpha = sum(t.replace("'", "").isalpha() for t in tokens) / len(tokens)
    lug = sum(t in LUG_FUNC or t.endswith("'") for t in tokens) / len(tokens)
    en = sum(t in EN_FUNC for t in tokens) / len(tokens)
    # Luganda words are vowel-final; English text has many consonant-final words
    vowel_final = sum(t[-1] in "aeiou'" for t in tokens if t) / len(tokens)
    return alpha > 0.85 and lug >= 0.08 and en < 0.05 and vowel_final > 0.7


def build_corpus():
    stats = {}
    sents = []

    raw_cc = [l for l in open(UNLAB / "cc100_lg.txt", encoding="utf-8", errors="ignore") if l.strip()]
    kept = [tokenize(l) for l in raw_cc]
    kept = [t for t in kept if is_luganda(t)]
    stats["cc100_nonempty_lines"] = len(raw_cc)
    stats["cc100_kept_lines"] = len(kept)
    stats["cc100_kept_ratio"] = len(kept) / len(raw_cc)
    sents += kept

    wiki = pd.read_parquet(UNLAB / "wiki_lg.parquet")
    wiki_lines = [tokenize(p) for doc in wiki["text"] for p in doc.split("\n")]
    wiki_kept = [t for t in wiki_lines if is_luganda(t)]
    stats["wiki_paragraphs"] = len(wiki_lines)
    stats["wiki_kept"] = len(wiki_kept)
    sents += wiki_kept

    train, dev, test = load_lug()
    news = [tokenize(s) for doc in train["input"] for s in re.split(r"(?<=[.!?])\s+", doc)]
    news = [t for t in news if len(t) >= 3]
    stats["masakhanews_train_sentences"] = len(news)
    sents += news

    # Leakage check: does any dev/test sentence (>= 8 tokens) appear verbatim in the unlabeled data?
    unlabeled_set = {" ".join(t) for t in kept + wiki_kept}
    held_out = [" ".join(tokenize(s)) for doc in pd.concat([dev, test])["input"]
                for s in re.split(r"(?<=[.!?])\s+", doc)]
    held_out = [s for s in held_out if len(s.split()) >= 8]
    stats["heldout_sentences_checked"] = len(held_out)
    stats["heldout_sentences_found_in_unlabeled"] = sum(s in unlabeled_set for s in held_out)

    stats["total_sentences"] = len(sents)
    stats["total_tokens"] = sum(len(s) for s in sents)
    return sents, stats


def main():
    sents, stats = build_corpus()
    print(json.dumps(stats, indent=2))
    # Skip-gram (sg=1) works better than CBOW for rare words, which dominate a morphologically
    # rich language (63% of training-set word types occur only once).
    model = Word2Vec(sentences=sents, vector_size=300, window=5, min_count=2, sg=1,
                     negative=10, epochs=10, workers=8, seed=42)
    MODELS_DIR.mkdir(exist_ok=True)
    model.wv.save(str(MODELS_DIR / "w2v_lug_300d.kv"))
    stats["vocab_size"] = len(model.wv)

    # Qualitative sanity check: nearest neighbours of topic words
    probes = ["omupiira", "ekkanisa", "ekirwadde", "akalulu", "ssente", "eddwaliro", "katonda", "ttiimu"]
    stats["nearest_neighbours"] = {w: [n for n, _ in model.wv.most_similar(w, topn=6)]
                                   for w in probes if w in model.wv}
    with open(RESULTS_DIR / "word2vec_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats["nearest_neighbours"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

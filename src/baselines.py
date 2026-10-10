"""Classical baselines: majority class, and TF-IDF features with linear classifiers.

Why these baselines:
  * Majority class gives the floor (what you get by always guessing "politics").
  * Word TF-IDF + Logistic Regression is the standard text-classification baseline and is the
    classical baseline reported in the MasakhaNEWS paper (Adelani et al., 2023).
  * Character n-gram TF-IDF tests a hypothesis specific to Luganda: the language is
    agglutinative, so one stem appears with many prefixes (e.g. "omukulembeze", "abakulembeze",
    "obukulembeze"). Char n-grams share evidence across these forms; whole-word features do not.

Hyperparameters (C, n-gram range) are chosen on the dev set only; the test set is used once.
Run: python src/baselines.py
"""
import json
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import LABEL2ID, MODELS_DIR, classical_clean, load_lug
from metrics import compute, full_report, save_run


def word_tfidf(ngram=(1, 2)):
    return TfidfVectorizer(preprocessor=classical_clean, ngram_range=ngram, min_df=2, sublinear_tf=True)


def char_tfidf(ngram=(2, 5)):
    return TfidfVectorizer(preprocessor=classical_clean, analyzer="char_wb", ngram_range=ngram,
                           min_df=2, sublinear_tf=True, max_features=200_000)


def candidates():
    """name -> list of (config description, pipeline) to search over on dev."""
    C_grid = [0.1, 1, 10, 100]
    return {
        "majority": [("most_frequent", Pipeline([("v", word_tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]))],
        "tfidf_word_nb": [(f"alpha={a}", Pipeline([("v", word_tfidf()), ("clf", ComplementNB(alpha=a))]))
                          for a in (0.01, 0.1, 0.5, 1.0)],
        "tfidf_word_lr": [(f"C={c},ngram={ng}", Pipeline([("v", word_tfidf(ng)),
                           ("clf", LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))]))
                          for c in C_grid for ng in [(1, 1), (1, 2)]],
        "tfidf_char_lr": [(f"C={c},ngram={ng}", Pipeline([("v", char_tfidf(ng)),
                           ("clf", LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))]))
                          for c in C_grid for ng in [(2, 4), (2, 5)]],
        "tfidf_word+char_svm": [(f"C={c}", Pipeline([("v", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
                                 ("clf", LinearSVC(C=c, class_weight="balanced"))]))
                                for c in (0.1, 0.5, 1, 5)],
    }


def main():
    train, dev, test = load_lug()
    y_tr, y_dev, y_te = (d["category"].map(LABEL2ID).values for d in (train, dev, test))
    summary = {}
    for name, cands in candidates().items():
        t0 = time.time()
        search = []
        for desc, pipe in cands:
            pipe.fit(train["input"], y_tr)
            search.append((compute(y_dev, pipe.predict(dev["input"]))["macro_f1"], desc, pipe))
        best_f1, best_desc, best = max(search, key=lambda s: s[0])
        pred = best.predict(test["input"])
        test_m = compute(y_te, pred)
        record = {"model": name, "family": "classical", "config": best_desc,
                  "dev_search": {d: f for f, d, _ in search}, "dev_macro_f1": best_f1,
                  "test": test_m, **full_report(y_te, pred), "train_seconds": time.time() - t0}
        save_run(name, record, preds=pred)
        summary[name] = {"config": best_desc, "dev_f1": round(best_f1, 4),
                         **{k: round(v, 4) for k, v in test_m.items()}}
        print(f"{name:22s} {best_desc:22s} dev F1={best_f1:.4f}  test F1={test_m['macro_f1']:.4f} acc={test_m['accuracy']:.4f}")
        if name == "tfidf_char_lr":  # saved so the web app can show a baseline side by side
            import joblib
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump(best, MODELS_DIR / "tfidf_char_lr.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    np.random.seed(42)
    main()

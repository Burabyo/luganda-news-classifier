"""Data loading and preprocessing for Luganda news topic classification.

Dataset: MasakhaNEWS (Adelani et al., 2023), Luganda split (source: gambuuze.ug).
Every model in this project reads data through this module so that all
experiments see exactly the same splits and the same cleaning.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

LABELS = ["business", "health", "politics", "religion", "sports"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Luganda uses the apostrophe heavily for elision (e.g. "n’obumalirivu" = "ne obumalirivu").
# Scraped text mixes U+2019, U+2018, U+0060 and U+0027, so we map them all to one symbol;
# otherwise the same word gets split into several vocabulary entries.
_APOSTROPHES = re.compile(r"[’‘`´ʼ]")
_QUOTES = re.compile(r"[“”„«»]")
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def normalize(text: str) -> str:
    """Light, lossless-ish normalisation shared by every model (case is preserved)."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("�", " ")          # broken bytes from scraping
    text = _URL.sub(" ", text)
    text = _APOSTROPHES.sub("'", text)
    text = _QUOTES.sub('"', text)
    return _WS.sub(" ", text).strip()


def classical_clean(text: str) -> str:
    """Extra cleaning for bag-of-words / RNN models: lowercase, keep letters, digits, apostrophe."""
    text = normalize(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return classical_clean(text).split()


def load_split(split: str, lang: str = "lug", field: str = "headline+text") -> pd.DataFrame:
    """Load one split ('train' | 'dev' | 'test') with a model-ready `input` column.

    field: 'headline', 'text' or 'headline+text' (default; the headline is prepended
    to the body because headlines are short, dense topic signals).
    """
    df = pd.read_csv(RAW_DIR / lang / f"{split}.tsv", sep="\t")
    df["headline"] = df["headline"].fillna("").map(normalize)
    df["text"] = df["text"].fillna("").map(normalize)
    if field == "headline":
        df["input"] = df["headline"]
    elif field == "text":
        df["input"] = df["text"]
    else:
        df["input"] = df["headline"] + ". " + df["text"]
    df["lang"] = lang
    return df.reset_index(drop=True)


def load_lug(field: str = "headline+text"):
    return tuple(load_split(s, "lug", field) for s in ("train", "dev", "test"))


def load_auxiliary(langs=("run", "swa"), field: str = "headline+text") -> pd.DataFrame:
    """Training data from related Bantu languages, restricted to the 5 Luganda labels.

    Used only in the cross-lingual transfer experiment; Luganda dev/test are never touched.
    """
    frames = [load_split("train", l, field) for l in langs]
    df = pd.concat(frames, ignore_index=True)
    return df[df["category"].isin(LABELS)].reset_index(drop=True)

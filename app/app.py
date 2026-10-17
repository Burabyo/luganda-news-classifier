"""Gradio web app: Luganda news topic classifier.

How the app connects to the ML model:
  1. The user's Luganda text is normalised with the SAME `normalize()` used in training.
  2. The fine-tuned Transformer (loaded from MODEL_ID: a local folder or a Hugging Face Hub repo)
     tokenises it into subwords, runs the encoder, and returns softmax probabilities for 5 topics.
  3. Word importance = occlusion: each word is removed in turn and we measure how much the
     probability of the predicted topic drops (bigger drop = more important word).
  4. The classical TF-IDF baseline (if present) is shown alongside for comparison.
  5. If the top probability is below CONF_THRESHOLD the app warns that the input may be outside
     the 5 trained topics or not Luganda (the model has no "other" class).

Run locally:  python app/app.py     (then open http://127.0.0.1:7860)
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
HUB_MODEL = "Burabyo/luganda-news-afroxlmr"   # fine-tuned weights published on the Hugging Face Hub
_local = [p for p in (HERE / "model", HERE.parent / "models" / "best") if (p / "config.json").exists()]
MODEL_ID = os.environ.get("MODEL_ID", str(_local[0]) if _local else HUB_MODEL)
BASELINE_PATH = HERE / "tfidf_char_lr.joblib"
if not BASELINE_PATH.exists():
    BASELINE_PATH = HERE.parent / "models" / "tfidf_char_lr.joblib"
CONF_THRESHOLD = 0.60
MAX_LEN = 256

LABEL_INFO = {
    "business": "Business / ebyobusuubuzi",
    "health": "Health / ebyobulamu",
    "politics": "Politics / ebyobufuzi",
    "religion": "Religion / eddiini",
    "sports": "Sports / emizannyo",
}

# --- same normalisation as src/data.py (duplicated so the Space is self-contained) ---
_APOS = re.compile(r"[’‘`´ʼ]")
_QUOTES = re.compile(r"[“”„«»]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).replace("�", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = _QUOTES.sub('"', _APOS.sub("'", text))
    return re.sub(r"\s+", " ", text).strip()


def classical_clean(text: str) -> str:  # needed to unpickle the baseline pipeline
    text = normalize(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]

baseline = None
try:
    import joblib
    import __main__
    __main__.classical_clean = classical_clean  # the pickled pipeline references this function
    sys.modules.setdefault("data", __main__)
    baseline = joblib.load(BASELINE_PATH)
except Exception as e:  # the app still works without the baseline
    print("Baseline not loaded:", e)


@torch.no_grad()
def probs_for(texts):
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    return torch.softmax(model(**enc).logits, -1).cpu()


def word_importance(text, label_idx, base_p, max_words=120):
    words = text.split()[:max_words]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    drops = []
    for i in range(0, len(variants), 32):
        p = probs_for(variants[i:i + 32])[:, label_idx]
        drops += (base_p - p).tolist()
    # HighlightedText expects (token, score) pairs; positive = supports the predicted topic
    m = max(1e-6, max(abs(d) for d in drops))
    return [(w + " ", round(d / m, 3)) for w, d in zip(words, drops)]


def classify(text, explain):
    text = normalize(text)
    if len(text.split()) < 3:
        return {}, "Please enter at least a sentence of Luganda text.", None, ""
    p = probs_for([text])[0]
    top = int(p.argmax())
    result = {LABEL_INFO[LABELS[i]]: float(p[i]) for i in range(len(LABELS))}
    conf = float(p[top])
    if conf < CONF_THRESHOLD:
        note = (f"⚠️ Low confidence ({conf:.0%}). The text may be about a topic outside the five "
                "trained categories, may not be Luganda, or may mix several topics. Treat this prediction with caution.")
    else:
        note = f"✅ Predicted topic: **{LABEL_INFO[LABELS[top]]}** with {conf:.0%} confidence."
    n_sub = len(tok(text)["input_ids"])
    if n_sub > MAX_LEN:
        note += f"\n\nℹ️ Text is {n_sub} subword tokens; only the first {MAX_LEN} are used."
    highlights = word_importance(text, top, float(p[top])) if explain else None
    base_txt = ""
    if baseline is not None:
        b = baseline.predict([text])[0]
        b = LABELS[b] if isinstance(b, (int,)) or str(b).isdigit() else str(b)
        base_txt = f"Classical baseline (TF-IDF char n-grams + logistic regression) predicts: **{b}**"
    return result, note, highlights, base_txt


EXAMPLES = [
    ["Ttiimu y'eggwanga The Cranes ewangudde omupiira gwayo ogw'okwetegekera empaka za AFCON nga ekubye ggoolo bbiri ku emu mu kisaawe e Namboole.", True],
    ["Minisitule y'ebyobulamu erabudde abantu okwegendereza ekirwadde kya kolera ekibaluseewo mu disitulikiti y'e Kasese, abalwadde basatu bafudde.", True],
    ["Ssaabasumba w'essaza ekkulu erya Kampala akulembeddemu Missa ey'okujjukira abajulizi e Namugongo, abakkiriza bangi beetabye mu kusaba.", True],
    ["Omubaka wa Palamenti akubye ebituli mu mbalirira y'eggwanga gye yayise ey'ekyejo, n'agamba nti gavumenti erina okulongoosa enteekateeka z'okulonda.", True],
    ["Abalimi b'emmwanyi mu Masaka basabye gavumenti okubayamba ku bbeeyi y'emmwanyi eyeeyongedde okukka ku katale k'ensi yonna.", True],
    ["Kabaka asabye abantu be okwettanira okulima n'okwekebeza omusaayi okumanya obulamu bwabwe.", True],
    ["The president addressed the nation on the new tax measures.", False],
]

with gr.Blocks(title="Luganda News Topic Classifier") as demo:
    gr.Markdown(
        "# 📰 Luganda News Topic Classifier\n"
        "Paste a Luganda news article or headline. A Transformer encoder fine-tuned on the Luganda part of "
        "[MasakhaNEWS](https://github.com/masakhane-io/masakhane-news) predicts its topic: "
        "**business, health, politics, religion or sports**.\n\n"
        "*Wandiika oba teekamu amawulire mu Luganda, tujja kukubuulira gakwata ku ki.*")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(lines=8, label="Luganda text", placeholder="Wandiika wano amawulire mu Luganda…")
            explain = gr.Checkbox(value=True, label="Show which words influenced the prediction (slower)")
            btn = gr.Button("Classify / Gerageza", variant="primary")
        with gr.Column(scale=2):
            out = gr.Label(num_top_classes=5, label="Topic probabilities")
            note = gr.Markdown()
            base = gr.Markdown()
    hl = gr.HighlightedText(label="Word importance (red = pushed towards the predicted topic, blue = against)",
                            combine_adjacent=False, show_legend=False)
    gr.Examples(EXAMPLES, inputs=[inp, explain])
    gr.Markdown(
        "**Limitations:** trained on 771 articles from a single news site (Gambuuze); only five topics, "
        "so text about anything else is forced into one of them; Luganda spelling variation and code-switching "
        "with English can reduce accuracy. Code, report and evaluation: see the GitHub repository.")
    btn.click(classify, [inp, explain], [out, note, hl, base])
    inp.submit(classify, [inp, explain], [out, note, hl, base])

if __name__ == "__main__":
    demo.launch()

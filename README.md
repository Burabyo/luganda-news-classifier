# Luganda News Topic Classification

**Text classification for an African language: automatically assigning Luganda news articles to one of five topics (business, health, politics, religion, sports).**

| | |
|---|---|
| 🌐 **Live demo** | `<HF_SPACE_URL>` |
| 🎥 **Demo video** | `<VIDEO_URL>` |
| 📄 **Report (PDF)** | [`report/report.pdf`](report/report.pdf) |
| 🤗 **Fine-tuned model** | `<HF_MODEL_URL>` |

---

## 1. Problem

Luganda (Ganda, ISO 639-3 `lug`) is a Bantu language spoken by about 11 million people in central Uganda, including Kampala. It is the most widely used Ugandan language in local media, yet it has almost no labelled NLP data. It is also **absent from the pretraining data of the most widely used African language models** (AfriBERTa covers 11 languages and AfroXLMR-base covers 17; Luganda is in neither).

This project builds and deploys a topic classifier for Luganda news and studies **what works when a language has only 771 labelled examples and was not seen by the pretrained model**:

* **RQ1:** Does African-centric pretraining (AfriBERTa, AfroXLMR) beat massively multilingual pretraining (XLM-R) and strong classical baselines for a language none of them saw?
* **RQ2:** Does exposure to Luganda in pretraining help? This is tested with (a) Serengeti, which includes Luganda, and (b) **our own language-adaptive fine-tuning (LAFT)** of AfroXLMR on 3.9M tokens of filtered Luganda web text.
* **RQ3:** Does labelled news from related Bantu languages (Kirundi, Swahili) help (cross-lingual transfer)?
* **RQ4:** Can parameter-efficient fine-tuning (LoRA) match full fine-tuning?

## 2. Data

| Resource | Use | Size | Licence |
|---|---|---|---|
| [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news) (Adelani et al., 2023), Luganda split | labelled train/dev/test | 771 / 110 / 223 articles, 5 classes | AFL-3.0 |
| MasakhaNEWS Kirundi + Swahili train | cross-lingual experiment (RQ3) | 1,117 + 1,658 articles (filtered to the 5 shared labels) | AFL-3.0 |
| [CC-100](https://data.statmt.org/cc-100/) `lg` (Conneau et al., 2020; Wenzek et al., 2020) | Word2Vec + LAFT (unlabeled) | 315k lines, **43% kept** after Luganda filtering | CC-100 terms |
| Wikipedia `20231101.lg` | Word2Vec + LAFT (unlabeled) | 4,048 articles | CC BY-SA |

Key data properties (from `results/eda.json`): politics makes up 45% of the training set (5.6:1 imbalance); articles average 232 tokens; 63% of word types occur only once; **42% of test word types never appear in training**. That last figure follows from Luganda's agglutinative morphology and makes vocabulary coverage the central challenge. All articles come from a single outlet (gambuuze.ug). No synthetic or LLM-generated data is used.

## 3. Methods

| Family | Models | Key idea |
|---|---|---|
| Classical baselines | Majority; TF-IDF + Naive Bayes / Logistic Regression / Linear SVM; word vs **character n-grams** | Char n-grams share evidence across inflected forms of the same stem |
| Static embeddings + RNN | **BiLSTM / BiGRU + additive attention**, random vs **Word2Vec (skip-gram) trained by us on Luganda** | Pretrained embeddings cut test OOV from 16.3% to 6.6% |
| Transformers (fine-tuned) | XLM-R-base, AfriBERTa-base, AfroXLMR-base, Serengeti | Pretrained multilingual encoders + classification head |
| Adaptation | **LAFT** (masked-LM on Luganda text), **cross-lingual training data**, **LoRA** | Different ways to bring in knowledge without more Luganda labels |

All hyperparameters and checkpoints are selected on the **dev** set, and the test set is scored once. Neural models are run with **3 seeds** (13, 42, 123) and reported as mean ± standard deviation. The primary metric is **macro-F1**, because the classes are imbalanced. Weighted-F1 is also reported so the numbers can be compared with the MasakhaNEWS paper.

## 4. Results

<!-- RESULTS_TABLE -->

Figures are in [`results/figures/`](results/figures). Full per-class reports and confusion matrices for every run are in [`results/runs/`](results/runs).

<!-- RESULTS_DISCUSSION -->

## 5. Repository layout

```
├── src/
│   ├── data.py              # loading, normalisation, splits (shared by every model)
│   ├── metrics.py           # macro/weighted F1, per-class report, confusion matrix, run logging
│   ├── eda.py               # dataset statistics + figures
│   ├── baselines.py         # majority, TF-IDF + NB / LR / SVM (word & char n-grams)
│   ├── word2vec_train.py    # CC-100/Wikipedia Luganda filtering, leakage check, skip-gram training
│   ├── rnn.py               # BiLSTM / BiGRU + attention (PyTorch)
│   ├── transformer_ft.py    # fine-tuning loop (full FT, LoRA, cross-lingual data, headline ablation)
│   ├── laft.py              # language-adaptive MLM fine-tuning on Luganda
│   ├── aggregate_results.py # experiment table + figure
│   └── error_analysis.py    # confusion matrices, error buckets, misclassified examples
├── app/app.py               # Gradio web app (Hugging Face Space)
├── scripts/download_models.sh
├── run_experiments.sh       # reproduces everything end-to-end
├── data/raw/                # MasakhaNEWS lug/run/swa TSVs (official splits)
├── results/                 # eda.json, runs/*.json, experiments.md/csv, figures/, error analysis
├── report/                  # final report (PDF + HTML source)
└── docs/                    # video script, viva preparation
```

## 6. Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# unlabeled Luganda text for Word2Vec / LAFT (~40 MB)
mkdir -p data/unlabeled
curl -L https://data.statmt.org/cc-100/lg.txt.xz | xz -d > data/unlabeled/cc100_lg.txt
curl -L -o data/unlabeled/wiki_lg.parquet https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.lg/train-00000-of-00001.parquet
bash run_experiments.sh          # ~6-8 h on an Apple M-series laptop (MPS); faster on a CUDA GPU
```

Individual stages can be run on their own, for example `python src/baselines.py` (about 1 minute) or `python src/transformer_ft.py --model Davlan/afro-xlmr-base --name afroxlmr_base --seeds 42`.

### Running the web app locally

```bash
python src/transformer_ft.py --model models/afroxlmr_base_laft_lug --name final --seeds 42 --save best
python app/app.py                # http://127.0.0.1:7860
```

## 7. Existing resources vs. own work

**Used (not ours):** the MasakhaNEWS dataset; CC-100 and Wikipedia text; the pretrained checkpoints XLM-R, AfriBERTa, AfroXLMR and Serengeti; and PyTorch, Hugging Face Transformers, PEFT, scikit-learn, gensim and Gradio.

**Implemented in this project:** text normalisation for Luganda orthography; a Luganda language filter for web text and a test-leakage check; Word2Vec training; the BiLSTM/BiGRU-attention model and its training loop; the transformer fine-tuning loop; language-adaptive pretraining; the cross-lingual, LoRA and input-ablation experiments; multi-seed evaluation, aggregation and error analysis; and the web application (occlusion-based word importance, low-confidence warning, baseline comparison).

## 8. References

See the reference list in the report. Key sources: Adelani et al. (2023) *MasakhaNEWS*; Ogueji et al. (2021) *AfriBERTa*; Alabi et al. (2022) *AfroXLMR / MAFT*; Adebara et al. (2023) *SERENGETI*; Conneau et al. (2020) *XLM-R*; Kreutzer et al. (2022) *Quality at a Glance*; Hu et al. (2022) *LoRA*; Mikolov et al. (2013) *Word2Vec*.

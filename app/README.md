---
title: Luganda News Topic Classifier
emoji: 📰
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Classify Luganda news into 5 topics (AfroXLMR fine-tuned)
---

# Luganda News Topic Classifier

Paste a Luganda news article or headline; the app predicts its topic — **business, health, politics, religion or sports** — with probabilities, word-level explanations (occlusion) and a TF-IDF baseline for comparison.

* Model: [AfroXLMR-base](https://huggingface.co/Davlan/afro-xlmr-base) fine-tuned on the Luganda split of [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news) → [`Burabyo/luganda-news-afroxlmr`](https://huggingface.co/Burabyo/luganda-news-afroxlmr)
* Test set (223 articles): 89.2% accuracy, 87.3 macro-F1 for this checkpoint (configuration mean over 3 seeds: 90.9% / 89.2)
* Code, experiments and report: https://github.com/Burabyo/luganda-news-classifier

**Limitations:** trained on 771 articles from a single news site (Gambuuze); only five topics (text about other subjects is still forced into one of them); confidence scores are over-confident.

---
language:
  - lg
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: Davlan/afro-xlmr-base
datasets:
  - masakhane/masakhanews
tags:
  - luganda
  - news
  - topic-classification
  - african-languages
  - transformers.js
metrics:
  - accuracy
  - f1
---

# Luganda news topic classifier (AfroXLMR-base)

Classifies a Luganda news article into one of five topics: **business, health, politics, religion, sports**.

* **Base model:** [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base). It was not pretrained on Luganda, but was adapted to 17 African languages, including several Bantu languages.
* **Training data:** Luganda split of [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news): 771 train and 110 dev articles from gambuuze.ug. Input is the headline + ". " + the article body.
* **Fine-tuning:** full fine-tuning, AdamW, lr 3e-5, batch 16, up to 8 epochs, 10% warm-up with linear decay, early stopping on dev macro-F1, max 256 subword tokens, seed 123.
* **Test results (223 articles):** this checkpoint: 89.2% accuracy, 87.3 macro-F1. Same configuration, mean of 3 seeds: 90.9% accuracy, 89.2 macro-F1. The strongest classical baseline (TF-IDF word+char SVM) gets 88.5 macro-F1.
* **Files:** `model.safetensors` (PyTorch, fp32) and `onnx/model_quantized.onnx` (8-bit dynamic quantisation, used by the in-browser demo).

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="Burabyo/luganda-news-afroxlmr", top_k=None)
clf("Ttiimu y'eggwanga The Cranes ewangudde omupiira gwayo mu kisaawe e Namboole.")
```

```js
import { pipeline } from "@huggingface/transformers";
const clf = await pipeline("text-classification", "Burabyo/luganda-news-afroxlmr", { dtype: "q8" });
```

## Limitations

The model was trained on a single news source with only five topics, so any input is forced into one of them; there is no "other" class. Some training labels are noisy or ambiguous, inputs are truncated to 256 tokens, and the confidence scores are over-confident. It has not been evaluated on social-media or code-switched text.

Demo: https://huggingface.co/spaces/Burabyo/luganda-news-classifier · Code and report: https://github.com/Burabyo/luganda-news-classifier

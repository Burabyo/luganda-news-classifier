"""Export the fine-tuned classifier to ONNX, quantise it to 8-bit, and check it still works.

Why: free Hugging Face hosting only allows *static* Spaces, so the deployed app runs the model in the
visitor's browser with Transformers.js (ONNX Runtime Web). Dynamic int8 quantisation stores each weight
matrix as 8-bit integers + a scale factor, shrinking the model ~4x (1.1 GB -> ~280 MB) for download.

Steps:
  1. load models/best (PyTorch, fp32)              -> reference predictions on the test set
  2. torch.onnx.export                             -> models/best/onnx/model.onnx (fp32)
  3. onnxruntime dynamic quantisation (QInt8)      -> models/best/onnx/model_quantized.onnx
  4. evaluate both with onnxruntime on the 223 test articles; report agreement, accuracy, macro-F1, size, latency
Output: results/onnx_eval.json
Run: python src/export_onnx.py
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import LABEL2ID, MODELS_DIR, RESULTS_DIR, load_lug
from metrics import compute

SRC = MODELS_DIR / "best"
ONNX_DIR = SRC / "onnx"
MAX_LEN = 256


def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()
    _, _, test = load_lug()
    y = test["category"].map(LABEL2ID).values
    texts = list(test["input"])

    # 1. reference (PyTorch, CPU)
    with torch.no_grad():
        ref = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            ref.append(model(**enc).logits.numpy())
    ref = np.concatenate(ref).argmax(1)

    # 2. export
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = ONNX_DIR / "model.onnx"
    dummy = tok(["Ttiimu ewangudde omupiira"], return_tensors="pt")
    model.config.return_dict = False
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                    "attention_mask": {0: "batch", 1: "sequence"},
                                    "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # 3. quantise weights to int8 (activations stay float, computed on the fly)
    q8 = ONNX_DIR / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8)

    # 4. evaluate
    out = {"pytorch_fp32": {**compute(y, ref), "size_mb": sum(f.stat().st_size for f in SRC.glob("*.safetensors")) / 1e6}}
    for name, path in [("onnx_fp32", fp32), ("onnx_int8", q8)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        preds, t0 = [], time.time()
        for t in texts:  # batch size 1, like the web app
            enc = tok([t], truncation=True, max_length=MAX_LEN, return_tensors="np")
            logits = sess.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0]
            preds.append(int(logits.argmax(1)[0]))
        preds = np.array(preds)
        out[name] = {**compute(y, preds), "size_mb": os.path.getsize(path) / 1e6,
                     "agreement_with_pytorch": float((preds == ref).mean()),
                     "cpu_ms_per_article": 1000 * (time.time() - t0) / len(texts)}
    with open(RESULTS_DIR / "onnx_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

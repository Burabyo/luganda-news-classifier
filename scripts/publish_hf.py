"""Publish the fine-tuned model and the static web app to the Hugging Face Hub.

  model repo  Burabyo/luganda-news-afroxlmr   <- models/best (+ onnx/model_quantized.onnx) + model card
  space       Burabyo/luganda-news-classifier <- space/index.html + space/README.md

Requires `hf auth login` first. Run: python scripts/publish_hf.py
"""
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / "models" / "best"
MODEL_REPO = "Burabyo/luganda-news-afroxlmr"
SPACE_REPO = "Burabyo/luganda-news-classifier"


def main():
    api = HfApi()
    # Training used 256 subword tokens; make the tokenizer truncate to the same length everywhere.
    cfg_path = BEST / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_max_length"] = 256
    cfg_path.write_text(json.dumps(cfg, indent=2))
    shutil.copy(ROOT / "space" / "MODEL_CARD.md", BEST / "README.md")

    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, folder_path=str(BEST), repo_type="model",
                      ignore_patterns=["onnx/model.onnx", "onnx/model.onnx.data", "*.bin", "training_args*"],
                      commit_message="Fine-tuned AfroXLMR-base for Luganda news topic classification (+ int8 ONNX)")
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=SPACE_REPO, folder_path=str(ROOT / "space"), repo_type="space",
                      allow_patterns=["index.html", "README.md"], commit_message="In-browser Luganda news classifier")
    print("model:", f"https://huggingface.co/{MODEL_REPO}")
    print("space:", f"https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()

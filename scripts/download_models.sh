#!/usr/bin/env bash
# Download the pretrained checkpoints used in the experiments into models/pretrained/<org>__<name>/.
# (Plain curl is used because it is resumable and robust on slow connections.)
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS="castorini/afriberta_base Davlan/afro-xlmr-base FacebookAI/xlm-roberta-base UBC-NLP/serengeti ${EXTRA_MODELS:-}"
for m in $MODELS; do
  dir="models/pretrained/${m/\//__}"; mkdir -p "$dir"
  files=$(curl -s "https://huggingface.co/api/models/$m" | python3 -c "
import json,sys
names=[s['rfilename'] for s in json.load(sys.stdin)['siblings']]
st=any(n=='model.safetensors' for n in names)
keep=[n for n in names if '/' not in n and (n.endswith(('.json','.model','.txt')) or n==('model.safetensors' if st else 'pytorch_model.bin'))]
print(' '.join(keep))")
  for f in $files; do
    echo "$m/$f"
    curl -sfL -C - --retry 10 --retry-all-errors "https://huggingface.co/$m/resolve/main/$f" -o "$dir/$f" || echo "FAILED $m/$f"
  done
done
du -sh models/pretrained/*

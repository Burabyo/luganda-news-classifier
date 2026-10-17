#!/usr/bin/env bash
# Reproduces every experiment in the report, in order. Results land in results/runs/*.json,
# then aggregate_results.py builds the tables and figures. Total time on an Apple M5 (MPS): ~8-10 h.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}

bash ../scripts/download_models.sh
$PY eda.py                       # dataset statistics + figures
$PY baselines.py                 # E0-E4: majority, NB, LR (word/char), SVM
$PY word2vec_train.py            # Luganda skip-gram embeddings from unlabeled text
$PY rnn.py                       # E5-E8: BiLSTM/BiGRU + attention, random vs Word2Vec

FT="$PY transformer_ft.py"
PT=../models/pretrained   # filled by scripts/download_models.sh
# E9-E12  RQ1: does African-centric pretraining help a language the model never saw?
$FT --model $PT/FacebookAI__xlm-roberta-base   --name xlmr_base
$FT --model $PT/FacebookAI__xlm-roberta-base   --name xlmr_base_stable --lr 1e-5 --epochs 15 --patience 5   # E10: stability check
$FT --model $PT/castorini__afriberta_base      --name afriberta_base --lr 5e-5
$FT --model $PT/Davlan__afro-xlmr-base         --name afroxlmr_base
# E13-E14 RQ2: does seeing Luganda in pretraining help? (models that include Luganda + our own LAFT)
$FT --model $PT/UBC-NLP__serengeti             --name serengeti --lr 5e-5
$PY laft.py
$FT --model ../models/afroxlmr_base_laft_lug --name afroxlmr_base_laft
# E15     RQ3: cross-lingual transfer from related Bantu languages (Kirundi, Swahili)
$FT --model $PT/Davlan__afro-xlmr-base         --name afroxlmr_base_aux --aux run,swa
# E16     RQ4: parameter-efficient fine-tuning (LoRA, r=16) vs full fine-tuning
$FT --model $PT/Davlan__afro-xlmr-base         --name afroxlmr_base_lora --lora --lr 5e-4 --epochs 12 --patience 4
# E17     input ablation: headline only
$FT --model $PT/Davlan__afro-xlmr-base         --name afroxlmr_base_headline --field headline --max_len 48

$PY aggregate_results.py
$PY error_analysis.py

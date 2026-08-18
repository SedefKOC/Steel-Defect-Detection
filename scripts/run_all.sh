#!/usr/bin/env bash
# CENG 476 - Severstal - GECELIK TAM CALISMA (Linux / macOS / Kaggle)
# Kullanim: bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
DATA="${1:-data}"

echo "[1/6] Hizli kontrol (smoke test)..."
python src/train.py --data_dir "$DATA" --limit 300 --epochs 2 --run_name smoke_test

echo "[2/6] Baseline model..."
python src/train.py --data_dir "$DATA" --run_name baseline \
  --pretrained 0 --use_bn 0 --decoder_dropout 0.0 --use_attention 0 \
  --weight_decay 0.0 --augment 0 --cls_loss_weight 0.0 --scheduler none --epochs 15

echo "[3/6] Final model..."
python src/train.py --data_dir "$DATA" --run_name final --epochs 30

echo "[4/6] Final degerlendirme..."
python src/evaluate.py --data_dir "$DATA" --run_name final

echo "[5/6] Baseline degerlendirme..."
python src/evaluate.py --data_dir "$DATA" --run_name baseline

echo "[6/6] Karsilastirma tablosu..."
python src/compare_runs.py --out_dir outputs

echo "BITTI. Sonuclar: outputs/final/ ve outputs/baseline/"

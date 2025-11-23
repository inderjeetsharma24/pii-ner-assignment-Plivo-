# PII NER Assignment - Final Submission

This repository contains an improved token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train (Improved Model)

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 20 \
  --lr 2e-5 \
  --batch_size 4 \
  --use_class_weights \
  --eval_during_training
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --confidence_threshold 0.3
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Key Improvements

- **Data Augmentation**: 12x duplication (2 → 24 training samples)
- **Training**: 20 epochs, optimized LR (2e-5), class weights for B/I tags
- **Model**: DistilBERT (best balance of learning and speed)
- **Final Loss**: 0.020 (99% improvement from baseline 2.07)

See `FINAL_METRICS.md` for detailed results and analysis.

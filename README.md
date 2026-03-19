# Pose Behavior Transformer

Benchmarking LSTM and transformer models for detecting human motor behaviors from pose sequences.

This repository implements a machine learning pipeline for temporal behavior recognition using OpenPose keypoints. The initial benchmark compares a multi-label LSTM baseline against a transformer encoder on pose-based sequence data.

## Current scope

- pose-based temporal modeling
- multi-label classification
- subject-independent evaluation with StratifiedGroupKFold
- labels: flapping, jumping

## Models

- LSTM baseline
- Transformer encoder

## Evaluation

- subject-independent StratifiedGroupKFold cross-validation
- per-label precision, recall, F1, specificity
- confusion matrices and classification reports

## Current benchmark

| Model       | F1 flap | F1 jump |
|-------------|---------|---------|
| LSTM        | 0.68    | 0.12    |
| Transformer | 0.71    | 0.18    |

## Repository structure

```text
src/
  models/
    lstm_model.py
    transformer_model.py
  training/
    cross_validation.py
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.training.train_lstm
python -m src.training.train_transformer
```

## Notes

This repository is currently being refactored from an earlier research codebase into a cleaner benchmark-oriented project structure.
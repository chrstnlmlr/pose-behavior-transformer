# Pose Behavior Transformer

Benchmarking LSTM and Transformer models for detecting human motor behaviors from pose sequences.

This repository implements a modular machine learning pipeline for temporal behavior recognition based on OpenPose keypoints. The project focuses on comparing sequence models under subject-independent evaluation conditions.

## Motivation

Manual behavioral coding is time-consuming and difficult to scale.  
This project explores whether sequence models can support or partially automate behavioral annotation from video-derived pose data.

## Task

- Input: sequences of pose keypoints (OpenPose)
- Output: multi-label classification of motor behaviors

Labels:
- flapping
- jumping

## Method

- sequence length: 15 frames
- feature representation: 2D keypoints
- multi-label classification (sigmoid outputs)
- strong class imbalance → undersampling strategy

## Models

- LSTM baseline (multi-layer)
- Transformer encoder (multi-head attention)

## Evaluation

- subject-independent splitting using StratifiedGroupKFold
- 3-fold cross-validation
- metrics:
  - precision
  - recall
  - F1-score
  - specificity
- evaluation performed per label

## Results

| Model       | F1 flap | Recall flap | F1 jump | Recall jump |
|-------------|---------|-------------|---------|-------------|
| LSTM        | 0.4692  | 0.3989      | 0.1985  | 0.1522      |
| Transformer | 0.4727  | 0.4344      | 0.2065  | 0.1794      |

### Interpretation

- Transformer slightly improves recall and F1 across both labels
- Jump behavior remains substantially harder to detect
- Results indicate potential benefits of attention-based models for temporal pose data

## Repository structure

```text
src/
  data/
    loader.py
    preprocessing.py
    prepare_data.py
  models/
    lstm_model.py
    transformer_model.py
  training/
    train_lstm.py
    train_transformer.py
    cross_validation.py
  evaluation/
    metrics.py
```

## Data

Raw data is not included due to size and privacy constraints.

To run the pipeline:

1. Place your dataset in:

```text
local_data/df_cleaned.csv
```

2. Prepare sequences:

```text
python -m src.data.prepare_data
```

3. Train models:

```text
python -m src.training.train_lstm
python -m src.training.train_transformer
```

## Requirements

```text
pip install -r requirements.txt
```

## Visualizations

Training curves and model comparison plots can be generated with:

```bash
python -m src.evaluation.plot_model_comparison
python -m src.evaluation.plot_training_curves
```

## Model comparison

![Model comparison](reports/figures/model_comparison.png)

## Status

This repository is part of an ongoing effort to build a clean, reproducible benchmark for pose-based behavior classification and to extend it toward more advanced sequence modeling approaches.
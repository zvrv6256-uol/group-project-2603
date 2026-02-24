# House Price Prediction with Feedforward ANN (+ Neighbourhood Features)

This repo implements:
1) A **baseline** feedforward neural network regressor on the California Housing dataset.
2) An **enhanced** model that adds simple **neighbourhood (graph-derived) features** built from geographic proximity.

> No Graph Neural Networks are used. A k-nearest-neighbours graph is only used to define neighbourhoods for feature construction.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run baseline + enhanced training:

```bash
python -m src.train --epochs 200 --batch-size 256 --k 8 --seed 42
```

Outputs (metrics + plots) are saved to `outputs/`.

## Key implementation notes

- Train/test split happens **before** building neighbourhood features.
- Neighbourhood features for **both train and test** are computed using **training labels only** (no leakage).
  - For each sample, we find its k nearest neighbours **among training samples** in (lat, lon).
  - We compute statistics from those neighbours: e.g. mean neighbour price, mean distance, etc.

## Files

- `src/data.py`: dataset loading + split + scaling
- `src/features.py`: KNN neighbourhood graph + feature construction (leakage-safe)
- `src/models.py`: PyTorch MLP regressor
- `src/train.py`: training loop, evaluation, plotting


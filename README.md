# House Price Prediction (California Housing) — ANN Baseline vs Neighbourhood Features (Optimized)

This repo contains a single runnable script:
- `house_price_ann_optimized.py`

It trains and evaluates:
1) **Baseline**: standard features → feedforward ANN (sklearn `MLPRegressor`)
2) **Enhanced**: standard features + neighbourhood (graph-derived) features → same ANN

## Key points (aligned with coursework brief)
- Regression with MSE loss.
- Feedforward neural network with non-linear activation (ReLU).
- Neighbourhood relationships are defined via k-nearest neighbours on (Latitude, Longitude).
- **No data leakage**: neighbourhood label statistics use **training labels only** for both train and test.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python house_price_ann_optimized.py --epochs 2000 --k 25 --seed 42
```

Notes:
- `--epochs` maps to `MLPRegressor(max_iter=...)`.
- The model uses `early_stopping=True` by default for stability.

## Optional
- Sweep k:
  ```bash
  python house_price_ann_optimized.py --sweep-k --k-list 5,10,25,50,100 --epochs 2000
  ```

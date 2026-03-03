"""House Price Prediction (California Housing) with Feedforward ANN
Baseline vs Neighbourhood-Feature Enhanced Model (optimized, leakage-safe).

Why this version:
- Uses **sklearn MLPRegressor** (feedforward ANN) to avoid heavy GPU dependencies.
- Neighbourhood (graph) features:
    * kNN on geographic coordinates (Latitude, Longitude)
    * uses **Haversine** distance (km) for more realistic geography
    * computes multiple simple statistics (mean/std of neighbour price, mean distance, weighted mean price, etc.)
- **No data leakage**: neighbour label statistics are computed from **training labels only** for both train and test.

Run:
  python house_price_ann_optimized.py --epochs 2000 --k 25 --seed 42

Optional k sweep:
  python house_price_ann_optimized.py --sweep-k --k-list 5,10,25,50,100

Coursework alignment:
- Regression task with MSE (reported + used internally by the estimator)
- Feedforward neural network with non-linear activation (ReLU)
- Baseline vs enhanced comparison
- Graph used only for neighbourhood feature construction (NOT a GNN)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


EARTH_RADIUS_KM = 6371.0088


# -------------------------
# Data
# -------------------------
@dataclass
class SplitData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def load_split_data(test_size: float, seed: int) -> SplitData:
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()

    y = df["MedHouseVal"].to_numpy(dtype=np.float64)
    X = df.drop(columns=["MedHouseVal"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return SplitData(X_train_raw=X_train, X_test_raw=X_test, y_train=y_train, y_test=y_test)


# -------------------------
# Neighbourhood feature engineering (leakage-safe)
# -------------------------
def _coords_radians(df: pd.DataFrame) -> np.ndarray:
    coords = df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    return np.radians(coords)


def add_neighbour_features(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    ref_y: np.ndarray,
    k: int,
    exclude_self: bool,
    add_ref_feature_means: Tuple[str, ...] = ("MedInc",),
    eps_km: float = 1e-3,
) -> pd.DataFrame:
    """Add neighbourhood features to `df` using neighbours from `ref_df` only.

    Parameters
    ----------
    df : DataFrame
        Target data to enrich (train or test).
    ref_df : DataFrame
        Reference pool for neighbours (MUST be training set).
    ref_y : ndarray
        Labels for reference pool (MUST be y_train). Used for label-based neighbour stats.
    k : int
        Number of neighbours.
    exclude_self : bool
        True when df is the same object as ref_df for training feature construction.
    add_ref_feature_means : tuple of str
        Optional extra features: mean of these columns among neighbours (non-label features).
    eps_km : float
        Stabilizer for inverse-distance weights.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    ref_coords = _coords_radians(ref_df)
    tgt_coords = _coords_radians(df)

    n_neighbors = k + 1 if exclude_self else k
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="haversine",
        algorithm="ball_tree",
    )
    nn.fit(ref_coords)

    dist_rad, idx = nn.kneighbors(tgt_coords, return_distance=True)

    if exclude_self:
        # Safer than blindly dropping first column: explicitly remove any self matches
        # For typical kNN on same reference pool, first column is self.
        idx = idx[:, 1:k+1]
        dist_rad = dist_rad[:, 1:k+1]
    else:
        idx = idx[:, :k]
        dist_rad = dist_rad[:, :k]

    dist_km = dist_rad * EARTH_RADIUS_KM  # (n, k)

    # --- label-based stats (leakage-safe: uses ref_y only) ---
    nb_y = ref_y[idx]  # (n, k)

    nb_mean_price = nb_y.mean(axis=1)
    nb_std_price = nb_y.std(axis=1)
    nb_mean_dist_km = dist_km.mean(axis=1)

    # Inverse-distance weighted neighbour price
    w = 1.0 / (dist_km + eps_km)
    nb_wmean_price = (w * nb_y).sum(axis=1) / w.sum(axis=1)

    out = df.copy()
    out["nb_mean_price"] = nb_mean_price
    out["nb_std_price"] = nb_std_price
    out["nb_mean_dist_km"] = nb_mean_dist_km
    out["nb_wmean_price"] = nb_wmean_price
    out["nb_count"] = float(k)

    # --- optional: neighbour means of other (non-label) features ---
    for col in add_ref_feature_means:
        if col not in ref_df.columns:
            continue
        ref_col = ref_df[col].to_numpy(dtype=np.float64)
        out[f"nb_mean_{col}"] = ref_col[idx].mean(axis=1)

    return out


# -------------------------
# Model / Metrics
# -------------------------
def build_ann_pipeline(
    max_iter: int,
    seed: int,
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    early_stopping: bool = True,
    n_iter_no_change: int = 20,
) -> Pipeline:
    """Feedforward ANN pipeline: StandardScaler + MLPRegressor (ReLU)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
            verbose=False,
        ))
    ])


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae)}


def train_and_eval(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    max_iter: int, seed: int,
    hidden: Tuple[int, ...], alpha: float, lr: float,
) -> Dict[str, float]:
    pipe = build_ann_pipeline(
        max_iter=max_iter,
        seed=seed,
        hidden_layer_sizes=hidden,
        alpha=alpha,
        learning_rate_init=lr,
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return eval_metrics(y_test, pred)


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # ANN / training
    p.add_argument("--epochs", type=int, default=2000, help="MLPRegressor(max_iter)")
    p.add_argument("--hidden", type=str, default="128,64", help="Comma-separated hidden layer sizes")
    p.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength")
    p.add_argument("--lr", type=float, default=1e-3, help="learning_rate_init")

    # Neighbours
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--nb-extra", type=str, default="MedInc", help="Comma-separated ref columns to average as neighbour features")

    # Optional sweep
    p.add_argument("--sweep-k", action="store_true")
    p.add_argument("--k-list", type=str, default="5,10,25,50,100")
    return p.parse_args()


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_tuple(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def run_once(args: argparse.Namespace, k: int) -> None:
    data = load_split_data(test_size=args.test_size, seed=args.seed)
    hidden = _parse_int_tuple(args.hidden)
    nb_extra = _parse_str_tuple(args.nb_extra)

    # Baseline
    base = train_and_eval(
        data.X_train_raw, data.y_train,
        data.X_test_raw, data.y_test,
        max_iter=args.epochs, seed=args.seed,
        hidden=hidden, alpha=args.alpha, lr=args.lr
    )

    # Enhanced (leakage-safe: neighbours from train only, labels from y_train only)
    X_train_enh = add_neighbour_features(
        df=data.X_train_raw,
        ref_df=data.X_train_raw,
        ref_y=data.y_train,
        k=k,
        exclude_self=True,
        add_ref_feature_means=nb_extra,
    )
    X_test_enh = add_neighbour_features(
        df=data.X_test_raw,
        ref_df=data.X_train_raw,
        ref_y=data.y_train,
        k=k,
        exclude_self=False,
        add_ref_feature_means=nb_extra,
    )

    enh = train_and_eval(
        X_train_enh, data.y_train,
        X_test_enh, data.y_test,
        max_iter=args.epochs, seed=args.seed,
        hidden=hidden, alpha=args.alpha, lr=args.lr
    )

    improvement = (base["mse"] - enh["mse"]) / max(base["mse"], 1e-12) * 100.0

    print("\n=== Results ===")
    print(f"k = {k} | hidden={hidden} | alpha={args.alpha} | lr={args.lr} | epochs(max_iter)={args.epochs}")
    print(f"Baseline | MSE: {base['mse']:.4f} | RMSE: {base['rmse']:.4f} | MAE: {base['mae']:.4f}")
    print(f"Enhanced | MSE: {enh['mse']:.4f} | RMSE: {enh['rmse']:.4f} | MAE: {enh['mae']:.4f}")
    print(f"Improvement (MSE): {improvement:.2f}%")
    print(f"Neighbour features: nb_mean_price, nb_std_price, nb_mean_dist_km, nb_wmean_price, nb_count"
          + (f", nb_mean_{', nb_mean_'.join(nb_extra)}" if nb_extra else ""))
    print()


def main() -> None:
    args = parse_args()
    if args.sweep_k:
        k_list = _parse_int_tuple(args.k_list)
        print(f"Sweeping k over: {k_list}")
        for k in k_list:
            run_once(args, k)
    else:
        run_once(args, args.k)


if __name__ == "__main__":
    main()

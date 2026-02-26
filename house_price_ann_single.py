"""House Price Prediction (California Housing) with a Feedforward ANN
Baseline vs Neighbourhood-Feature Enhanced Model (leakage-safe).

Usage:
  pip install numpy pandas scikit-learn torch tqdm matplotlib
  python house_price_ann_single.py --epochs 200 --batch-size 256 --k 8 --seed 42

Notes:
- This is NOT a GNN. A kNN index is used only to define neighbourhoods for feature construction.
- Neighbourhood features are computed using TRAINING labels only (no test label leakage):
    * For each sample (train/test), neighbours are selected among TRAIN samples only using (lat, lon).
    * Neighbour label statistics use y_train only.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def mse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean((y_pred.view(-1) - y_true.view(-1)) ** 2).item()


@torch.no_grad()
def rmse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_pred.view(-1) - y_true.view(-1)) ** 2)).item())


@torch.no_grad()
def mae_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred.view(-1) - y_true.view(-1))).item()


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km). Inputs can be numpy arrays."""
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# -------------------------
# Data
# -------------------------
@dataclass
class Bundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    coords_train: np.ndarray  # (lat, lon) unscaled
    coords_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def load_data(test_size: float, seed: int) -> Bundle:
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()

    y = df["MedHouseVal"].to_numpy(dtype=np.float32)
    feature_names = [c for c in df.columns if c != "MedHouseVal"]
    X = df[feature_names].to_numpy(dtype=np.float32)

    # coords (Latitude, Longitude) are part of X (unscaled here)
    try:
        lat_idx = feature_names.index("Latitude")
        lon_idx = feature_names.index("Longitude")
    except ValueError as e:
        raise RuntimeError("Expected Latitude/Longitude in dataset features.") from e

    coords = X[:, [lat_idx, lon_idx]].astype(np.float32)

    # Split FIRST (important for leakage control)
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=test_size, random_state=seed
    )

    # Standardize features using train only
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return Bundle(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        coords_train=coords_train.astype(np.float32),
        coords_test=coords_test.astype(np.float32),
        feature_names=feature_names,
        scaler=scaler,
    )


# -------------------------
# Neighbourhood features (leakage-safe)
# -------------------------
def compute_neighbour_features(
    coords_train: np.ndarray,
    coords_test: np.ndarray,
    y_train: np.ndarray,
    k: int = 8,
    use_haversine: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute neighbourhood-derived features for train and test.

    Neighbours are selected among TRAIN points only.
    Label-derived features use y_train only (no test labels are used).
    """
    # +1 for train points because their nearest neighbour is themselves
    nn_index = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_index.fit(coords_train)

    # Train neighbours
    _, idx_tr = nn_index.kneighbors(coords_train, return_distance=True)
    idx_tr_k = idx_tr[:, 1 : k + 1]  # drop self

    # Test neighbours (neighbours among train only, so just take first k)
    _, idx_te = nn_index.kneighbors(coords_test, return_distance=True)
    idx_te_k = idx_te[:, :k]

    # Distances (optional, in km for interpretability)
    if use_haversine:
        tr_lat = coords_train[:, 0:1]
        tr_lon = coords_train[:, 1:2]
        nb_tr = coords_train[idx_tr_k]  # (n_train, k, 2)
        dist_tr = haversine_km(tr_lat, tr_lon, nb_tr[:, :, 0], nb_tr[:, :, 1]).astype(np.float32)

        te_lat = coords_test[:, 0:1]
        te_lon = coords_test[:, 1:2]
        nb_te = coords_train[idx_te_k]
        dist_te = haversine_km(te_lat, te_lon, nb_te[:, :, 0], nb_te[:, :, 1]).astype(np.float32)
    else:
        # Less interpretable; degrees distance
        dist_tr, _ = nn_index.kneighbors(coords_train, return_distance=True)
        dist_te, _ = nn_index.kneighbors(coords_test, return_distance=True)
        dist_tr = dist_tr[:, 1 : k + 1].astype(np.float32)
        dist_te = dist_te[:, :k].astype(np.float32)

    # Label stats (training labels only)
    nb_y_tr = y_train[idx_tr_k]  # (n_train, k)
    nb_y_te = y_train[idx_te_k]  # (n_test, k)

    # Example neighbour features (edit/add more if you want)
    f_tr = np.concatenate(
        [
            nb_y_tr.mean(axis=1, keepdims=True).astype(np.float32),  # mean neighbour price
            nb_y_tr.std(axis=1, keepdims=True).astype(np.float32),   # std neighbour price
            dist_tr.mean(axis=1, keepdims=True).astype(np.float32),  # mean neighbour distance
            np.full((coords_train.shape[0], 1), float(k), dtype=np.float32),  # neighbour count
        ],
        axis=1,
    )

    f_te = np.concatenate(
        [
            nb_y_te.mean(axis=1, keepdims=True).astype(np.float32),
            nb_y_te.std(axis=1, keepdims=True).astype(np.float32),
            dist_te.mean(axis=1, keepdims=True).astype(np.float32),
            np.full((coords_test.shape[0], 1), float(k), dtype=np.float32),
        ],
        axis=1,
    )

    names = ["nb_mean_price", "nb_std_price", "nb_mean_dist", "nb_count"]
    return f_tr, f_te, names


# -------------------------
# Model
# -------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 32), dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# -------------------------
# Train / Eval
# -------------------------
def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        ys.append(yb)
        preds.append(yhat)
    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(preds, dim=0)
    return {"mse": mse_torch(y_pred, y_true), "rmse": rmse_torch(y_pred, y_true), "mae": mae_torch(y_pred, y_true)}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dims: Tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in tqdm(range(epochs), desc="Training", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()

    return evaluate(model, test_loader, device=device)


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden-dims", type=str, default="64,32")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--no-haversine", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())

    bundle = load_data(test_size=args.test_size, seed=args.seed)

    # Baseline
    baseline = train_model(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # Neighbour features (leakage-safe)
    nb_tr, nb_te, nb_names = compute_neighbour_features(
        coords_train=bundle.coords_train,
        coords_test=bundle.coords_test,
        y_train=bundle.y_train,
        k=args.k,
        use_haversine=(not args.no_haversine),
    )

    X_train_enh = np.concatenate([bundle.X_train, nb_tr], axis=1).astype(np.float32)
    X_test_enh = np.concatenate([bundle.X_test, nb_te], axis=1).astype(np.float32)

    enhanced = train_model(
        X_train_enh,
        bundle.y_train,
        X_test_enh,
        bundle.y_test,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n=== Results ===")
    print(f"Baseline | Test MSE: {baseline['mse']:.4f} | RMSE: {baseline['rmse']:.4f} | MAE: {baseline['mae']:.4f}")
    print(f"Enhanced | Test MSE: {enhanced['mse']:.4f} | RMSE: {enhanced['rmse']:.4f} | MAE: {enhanced['mae']:.4f}")
    improvement = (baseline["mse"] - enhanced["mse"]) / max(baseline["mse"], 1e-12) * 100.0
    print(f"Improvement (MSE): {improvement:.2f}%")
    print(f"Neighbour features added: {nb_names} (k={args.k})\n")


if __name__ == "__main__":
    main()

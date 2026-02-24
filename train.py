from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import load_california_housing
from .features import compute_neighbour_features
from .models import MLPRegressor
from .utils import set_seed, ensure_dir, mse_torch, rmse_torch, mae_torch


@dataclass
class TrainResult:
    test_mse: float
    test_rmse: float
    test_mae: float


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float()
    Xte = torch.from_numpy(X_test).float()
    yte = torch.from_numpy(y_test).float()

    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys = []
    preds = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        ys.append(yb)
        preds.append(yhat)
    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(preds, dim=0)
    return {
        "mse": mse_torch(y_pred, y_true),
        "rmse": rmse_torch(y_pred, y_true),
        "mae": mae_torch(y_pred, y_true),
    }


def train_one(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> TrainResult:
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(1, epochs + 1), desc="Training", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()

    metrics = evaluate(model, test_loader, device=device)
    return TrainResult(test_mse=metrics["mse"], test_rmse=metrics["rmse"], test_mae=metrics["mae"])


def main() -> None:
    parser = argparse.ArgumentParser(description="House Price ANN: baseline vs neighbourhood features")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dims", type=str, default="64,32", help="Comma-separated, e.g. 128,64,32")
    parser.add_argument("--k", type=int, default=8, help="k for kNN neighbourhood features")
    parser.add_argument("--no-haversine", action="store_true", help="Use euclidean distance instead of haversine km")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    bundle = load_california_housing(test_size=args.test_size, seed=args.seed)

    # --- Baseline ---
    baseline = train_one(
        X_train=bundle.X_train, y_train=bundle.y_train,
        X_test=bundle.X_test, y_test=bundle.y_test,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # --- Neighbourhood features (leakage-safe) ---
    nb = compute_neighbour_features(
        coords_train=bundle.coords_train,
        coords_test=bundle.coords_test,
        y_train=bundle.y_train,              # training labels ONLY
        X_train_scaled=bundle.X_train,       # (kept in signature if you extend features)
        k=args.k,
        use_haversine=(not args.no_haversine),
    )

    X_train_enh = np.concatenate([bundle.X_train, nb.train_feats], axis=1).astype(np.float32)
    X_test_enh = np.concatenate([bundle.X_test, nb.test_feats], axis=1).astype(np.float32)

    enhanced = train_one(
        X_train=X_train_enh, y_train=bundle.y_train,
        X_test=X_test_enh, y_test=bundle.y_test,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # Print summary
    print("\n=== Results ===")
    print(f"Baseline  | Test MSE: {baseline.test_mse:.4f} | RMSE: {baseline.test_rmse:.4f} | MAE: {baseline.test_mae:.4f}")
    print(f"Enhanced  | Test MSE: {enhanced.test_mse:.4f} | RMSE: {enhanced.test_rmse:.4f} | MAE: {enhanced.test_mae:.4f}")
    improvement = (baseline.test_mse - enhanced.test_mse) / max(baseline.test_mse, 1e-12) * 100.0
    print(f"Improvement (MSE): {improvement:.2f}%\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    # Scaled feature matrices
    X_train: np.ndarray
    X_test: np.ndarray
    # Labels
    y_train: np.ndarray
    y_test: np.ndarray
    # Unscaled coordinates used for neighbourhood graph
    coords_train: np.ndarray  # shape (n_train, 2) -> [lat, lon]
    coords_test: np.ndarray   # shape (n_test, 2)
    # Feature names
    feature_names: list[str]
    # scaler (for reference / reproducibility)
    scaler: StandardScaler


def load_california_housing(
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    """
    Loads California Housing dataset from scikit-learn.
    Uses 'Latitude' and 'Longitude' columns as geographic coordinates for neighbourhood graph.
    """
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()

    # Target
    y = df["MedHouseVal"].to_numpy(dtype=np.float32)

    # Input features
    feature_names = [c for c in df.columns if c != "MedHouseVal"]
    X = df[feature_names].to_numpy(dtype=np.float32)

    # Coordinates are part of features: Latitude, Longitude
    try:
        lat_idx = feature_names.index("Latitude")
        lon_idx = feature_names.index("Longitude")
    except ValueError as e:
        raise RuntimeError("Expected Latitude/Longitude in feature set.") from e

    coords = X[:, [lat_idx, lon_idx]].astype(np.float32)

    # Split first (IMPORTANT for leakage control)
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=test_size, random_state=seed
    )

    # Standardize features using train only
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return DatasetBundle(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        coords_train=coords_train.astype(np.float32),
        coords_test=coords_test.astype(np.float32),
        feature_names=feature_names,
        scaler=scaler,
    )

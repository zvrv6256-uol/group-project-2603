from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class NeighbourFeatures:
    train_feats: np.ndarray
    test_feats: np.ndarray
    feature_names: list[str]


def build_knn_index(
    coords_train: np.ndarray,
    k: int,
) -> NearestNeighbors:
    """
    Builds a kNN index on training coordinates.
    Coordinates are [Latitude, Longitude].
    """
    # +1 because the nearest neighbor of a training point is itself.
    # We'll remove self later for train features.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords_train)
    return nn


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Compute great-circle distances (km) between points.
    Accepts numpy arrays / floats.
    """
    R = 6371.0088  # mean earth radius in km
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c


def compute_neighbour_features(
    coords_train: np.ndarray,
    coords_test: np.ndarray,
    y_train: np.ndarray,
    X_train_scaled: np.ndarray,
    k: int = 8,
    use_haversine: bool = True,
) -> NeighbourFeatures:
    """
    Leakage-safe neighbourhood features:
      - Neighbours are always selected among TRAIN points only.
      - Any feature involving labels uses y_train only.
    """
    nn = build_knn_index(coords_train, k=k)

    # Query neighbours for train points
    # indices_train shape: (n_train, k+1)
    dists_train, indices_train = nn.kneighbors(coords_train, return_distance=True)

    # Remove self neighbour for train: the first index is itself with dist 0 (for euclidean).
    # For robustness, drop the first column.
    indices_train_k = indices_train[:, 1:k+1]

    # Query neighbours for test points: (n_test, k+1) but for test there is no "self" in train,
    # so we can just take first k (not k+1). However our index is built with k+1.
    dists_test, indices_test = nn.kneighbors(coords_test, return_distance=True)
    indices_test_k = indices_test[:, :k]

    # Distances: use haversine km for interpretability if desired
    if use_haversine:
        # Compute distances in km for train
        tr_lat = coords_train[:, 0:1]
        tr_lon = coords_train[:, 1:2]
        nb_tr = coords_train[indices_train_k]  # (n_train, k, 2)
        dist_tr_km = haversine_km(tr_lat, tr_lon, nb_tr[:, :, 0], nb_tr[:, :, 1]).astype(np.float32)

        # Compute distances in km for test
        te_lat = coords_test[:, 0:1]
        te_lon = coords_test[:, 1:2]
        nb_te = coords_train[indices_test_k]
        dist_te_km = haversine_km(te_lat, te_lon, nb_te[:, :, 0], nb_te[:, :, 1]).astype(np.float32)
    else:
        # Use euclidean distances from sklearn (in degrees), less interpretable
        dist_tr_km = dists_train[:, 1:k+1].astype(np.float32)
        dist_te_km = dists_test[:, :k].astype(np.float32)

    # Neighbour label stats (training labels only)
    nb_y_train = y_train[indices_train_k]  # (n_train, k)
    nb_y_test = y_train[indices_test_k]    # (n_test, k) -- using y_train ONLY

    # Neighbour feature examples (add/modify freely)
    # 1) mean neighbour price
    mean_nb_price_train = nb_y_train.mean(axis=1, keepdims=True).astype(np.float32)
    mean_nb_price_test = nb_y_test.mean(axis=1, keepdims=True).astype(np.float32)

    # 2) std neighbour price
    std_nb_price_train = nb_y_train.std(axis=1, keepdims=True).astype(np.float32)
    std_nb_price_test = nb_y_test.std(axis=1, keepdims=True).astype(np.float32)

    # 3) mean distance to neighbours
    mean_nb_dist_train = dist_tr_km.mean(axis=1, keepdims=True).astype(np.float32)
    mean_nb_dist_test = dist_te_km.mean(axis=1, keepdims=True).astype(np.float32)

    # 4) neighbour count (always k, but included for completeness / if you later use radius graph)
    nb_count_train = np.full((coords_train.shape[0], 1), float(k), dtype=np.float32)
    nb_count_test = np.full((coords_test.shape[0], 1), float(k), dtype=np.float32)

    train_feats = np.concatenate(
        [mean_nb_price_train, std_nb_price_train, mean_nb_dist_train, nb_count_train],
        axis=1
    )
    test_feats = np.concatenate(
        [mean_nb_price_test, std_nb_price_test, mean_nb_dist_test, nb_count_test],
        axis=1
    )

    feat_names = [
        "nb_mean_price",
        "nb_std_price",
        "nb_mean_dist_km",
        "nb_count",
    ]

    return NeighbourFeatures(train_feats=train_feats, test_feats=test_feats, feature_names=feat_names)

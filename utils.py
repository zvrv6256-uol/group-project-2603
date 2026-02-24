from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may reduce performance a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def mse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean((y_pred.view(-1) - y_true.view(-1)) ** 2).item()


@torch.no_grad()
def rmse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_pred.view(-1) - y_true.view(-1)) ** 2)).item())


@torch.no_grad()
def mae_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred.view(-1) - y_true.view(-1))).item()

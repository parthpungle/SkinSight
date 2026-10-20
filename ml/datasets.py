"""Dataset over the split manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.preprocessing import load_rgb


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str | Path, transform: Callable):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = load_rgb(self.root / row["image_path"])
        return self.transform(img), int(row["label"])


def load_split(manifest_path: str | Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    return df[df["split"] == split].reset_index(drop=True)

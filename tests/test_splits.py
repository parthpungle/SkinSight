from pathlib import Path

import pandas as pd
import pytest

from ml.config import data_path, load_config
from ml.data.build_manifest import map_labels
from ml.data.create_splits import split_by_group


def _synthetic(n_lesions: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n_lesions):
        label = int(i % 5 == 0)
        for j in range(1 + i % 3):  # 1-3 images per lesion
            rows.append({"lesion_id": f"L{i}", "image_id": f"I{i}_{j}", "label": label})
    return pd.DataFrame(rows)


def test_split_has_no_group_overlap():
    out = split_by_group(_synthetic(), "lesion_id", "label", 0.15, 0.15, seed=0)
    assert out["split"].notna().all()
    assert (out.groupby("lesion_id")["split"].nunique() == 1).all()


def test_split_is_reproducible():
    a = split_by_group(_synthetic(), "lesion_id", "label", 0.15, 0.15, seed=7)
    b = split_by_group(_synthetic(), "lesion_id", "label", 0.15, 0.15, seed=7)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_split_keeps_both_classes_everywhere():
    out = split_by_group(_synthetic(), "lesion_id", "label", 0.15, 0.15, seed=0)
    assert (out.groupby("split")["label"].nunique() == 2).all()


def test_map_labels_rejects_unknown_dx():
    df = pd.DataFrame({"dx": ["mel", "nv", "weird"]})
    with pytest.raises(ValueError):
        map_labels(df, ["mel"], ["nv"])


def test_map_labels_positive_is_malignant():
    df = map_labels(pd.DataFrame({"dx": ["mel", "nv", "bcc"]}), ["mel", "bcc"], ["nv"])
    assert df["label"].tolist() == [1, 0, 1]


def test_real_split_has_no_lesion_leakage():
    cfg = load_config()
    path = data_path(cfg, "splits") / "manifest_split.csv"
    if not Path(path).is_file():
        pytest.skip("real split not generated yet")
    df = pd.read_csv(path)
    assert df["split"].isin(["train", "val", "test"]).all()
    assert (df.groupby("lesion_id")["split"].nunique() == 1).all()

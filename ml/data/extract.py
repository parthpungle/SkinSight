"""Extract the HAM10000 zips into flat image and mask folders.

Usage:
    python -m ml.data.extract

Only members named like ISIC_0000000.jpg / ISIC_0000000_segmentation.png are written, by basename,
so archive paths can never escape the target directories.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from ml.config import data_path, load_config

IMAGE_RE = re.compile(r"^ISIC_\d{7}\.jpg$")
MASK_RE = re.compile(r"^ISIC_\d{7}_segmentation\.png$")


def extract(zip_path: Path, target: Path, pattern: re.Pattern) -> int:
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not pattern.match(name):
                continue
            out = target / name
            if not out.exists() or out.stat().st_size != info.file_size:
                out.write_bytes(zf.read(info))
            n += 1
    return n


def main() -> None:
    cfg = load_config()
    zips = data_path(cfg, "raw_zips")
    n_img = sum(extract(zips / f"HAM10000_images_part_{i}.zip", data_path(cfg, "images"), IMAGE_RE) for i in (1, 2))
    n_mask = extract(zips / "HAM10000_segmentations_lesion_tschandl.zip", data_path(cfg, "masks"), MASK_RE)
    print(f"images: {n_img}, masks: {n_mask}")


if __name__ == "__main__":
    main()

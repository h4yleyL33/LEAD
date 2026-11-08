"""Fetch the LEAD1.0-small dataset (real, annotated) into data/raw/lead/.

The public reference subset is published by the dataset authors and is the primary
data source for this project. The Kaggle competition split requires credentials
that are not available here; if they later appear, `--kaggle` fetches that instead.

Usage:
    python scripts/fetch_data.py            # public LEAD1.0-small
    python scripts/fetch_data.py --kaggle   # competition split (needs ~/.kaggle/kaggle.json)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import LEAD_RAW_CSV, LEAD_RAW_ZIP, RAW_DIR  # noqa: E402

LEAD_SMALL_URL = (
    "https://raw.githubusercontent.com/samy101/lead-dataset/main/data/lead1.0-small.zip"
)
# SHA-256 of the archive as downloaded on 2026-11-09; verified after download.
LEAD_SMALL_SHA256 = "b0e34c66c0c4ce780b7bb19b199d33e19210675f16be102a5ac32d6b22e39d8f"
# SHA-256 of the extracted CSV (integrity of the exact file used for all results).
LEAD_SMALL_CSV_SHA256 = "a161537ff06230a50c2d2272e026641c4cbbbadc938af28406b3d6881162b32c"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_public(force: bool = False) -> Path:
    if LEAD_RAW_CSV.exists() and not force:
        print(f"[skip] already present: {LEAD_RAW_CSV} ({LEAD_RAW_CSV.stat().st_size/1e6:.1f} MB)")
        return LEAD_RAW_CSV

    LEAD_RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not LEAD_RAW_ZIP.exists() or force:
        print(f"[get ] {LEAD_SMALL_URL}")
        urllib.request.urlretrieve(LEAD_SMALL_URL, LEAD_RAW_ZIP)  # noqa: S310

    digest = sha256(LEAD_RAW_ZIP)
    print(f"[hash] sha256={digest}")
    print(f"[size] {LEAD_RAW_ZIP.stat().st_size/1e6:.2f} MB")
    if LEAD_SMALL_SHA256 and digest != LEAD_SMALL_SHA256:
        raise SystemExit(
            f"checksum mismatch: expected {LEAD_SMALL_SHA256}, got {digest}. "
            "The upstream file may have changed; re-audit before trusting results."
        )

    with zipfile.ZipFile(LEAD_RAW_ZIP) as zf:
        names = zf.namelist()
        print(f"[zip ] members: {names}")
        zf.extractall(LEAD_RAW_ZIP.parent)

    if not LEAD_RAW_CSV.exists():
        raise SystemExit(f"extraction did not produce {LEAD_RAW_CSV}")
    print(f"[ok  ] {LEAD_RAW_CSV} ({LEAD_RAW_CSV.stat().st_size/1e6:.1f} MB)")
    return LEAD_RAW_CSV


def fetch_kaggle() -> None:
    """Download the competition split if credentials are configured."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        raise SystemExit(
            "kaggle package not installed. pip install kaggle, place kaggle.json in "
            "~/.kaggle/, accept the competition rules, then re-run."
        )
    api = KaggleApi()
    api.authenticate()
    dest = RAW_DIR / "lead"
    dest.mkdir(parents=True, exist_ok=True)
    api.competition_download_files("energy-anomaly-detection", path=str(dest), quiet=False)
    for zf in dest.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
    print(f"[ok  ] competition files in {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="fetch competition split instead")
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args()
    if args.kaggle:
        fetch_kaggle()
    else:
        fetch_public(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Compute 64-bit perceptual hashes (pHash) for images and write them to metadata/images.csv.

Usage:
  python preprocess/dup/phash.py \
    --repo-root . \
    --metadata metadata/images.csv \
    --out-col phash64 \
    --backup-csv
"""

import argparse
import logging
import os
import sys
import tempfile

import pandas as pd
from PIL import Image, UnidentifiedImageError
import imagehash

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def s(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and x != x: return ""
    return str(x).strip()

def truthy(x) -> bool:
    return s(x).lower() in {"true","1","yes","y","t"}

def compute_phash_hex(img_path: str) -> str | None:
    try:
        with Image.open(img_path) as im:
            im.seek(0)
            return str(imagehash.phash(im.convert("RGB")))
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None

def atomic_write_csv(df: pd.DataFrame, csv_path: str, backup: bool):
    if backup and os.path.exists(csv_path):
        bak = f"{csv_path}.bak"
        pd.read_csv(csv_path).to_csv(bak, index=False)
        logging.info("Backed up CSV to %s", bak)
    dirn = os.path.dirname(csv_path) or "."
    fd, tmp = tempfile.mkstemp(prefix="images.", suffix=".csv.tmp", dir=dirn); os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Compute perceptual hash (pHash) for images and write to metadata CSV.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--metadata", default="metadata/images.csv")
    ap.add_argument("--out-col", default="phash64")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup-csv", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    csv_path = args.metadata if os.path.isabs(args.metadata) else os.path.join(repo, args.metadata)
    if not os.path.exists(csv_path):
        logging.error("Metadata CSV not found: %s", csv_path); sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    if args.out_col not in df.columns: df[args.out_col] = ""

    n_total = len(df); n_done = n_skip = n_missing = n_err = 0
    n_used_raw = n_used_resized = 0

    for i in range(n_total):
        row = df.iloc[i]
        if s(row.get(args.out_col)) and not args.overwrite:
            n_skip += 1; continue

        img_path = None
        if truthy(row.get("is_nsfw")):
            pr = s(row.get("path_raw"))
            if pr: img_path = pr if os.path.isabs(pr) else os.path.join(repo, pr); n_used_raw += int(os.path.exists(img_path))
        else:
            op = s(row.get("out_path"))
            if op and os.path.isabs(op) and os.path.exists(op):
                img_path = op; n_used_resized += 1
            else:
                pr = s(row.get("path_resized"))
                if pr:
                    cand = pr if os.path.isabs(pr) else os.path.join(repo, pr)
                    if os.path.exists(cand): img_path = cand; n_used_resized += 1

        if not img_path:
            n_missing += 1; continue

        ph = compute_phash_hex(img_path)
        if ph is None:
            n_err += 1; continue

        df.at[i, args.out_col] = ph
        n_done += 1

    atomic_write_csv(df, csv_path, backup=args.backup_csv)
    logging.info("pHash complete: total=%d wrote=%d skipped=%d missing=%d errors=%d used_raw=%d used_resized=%d",
                 n_total, n_done, n_skip, n_missing, n_err, n_used_raw, n_used_resized)

if __name__ == "__main__":
    main()

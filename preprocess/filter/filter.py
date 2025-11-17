"""
Filter by aesthetic & relevance, copy top-N to data_clean/base, and write base.csv.

Usage:
  python preprocess/filter/filter.py \
    --repo-root . \
    --metadata metadata/images.csv \
    --out-base-dir data_clean/base \
    --out-base-csv metadata/base.csv \
    --top-n 16000 \
    --min-relevance 60 \
    --min-aesthetic 60
"""

import argparse
import csv
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List

import pandas as pd
from PIL import Image
from tqdm import tqdm


def resolve_resized_path(row: pd.Series, repo_root: Path) -> Optional[Path]:
    p = row.get("path_resized")
    if isinstance(p, str) and p.strip():
        pth = Path(p)
        if not pth.is_absolute():
            pth = repo_root / pth
        if pth.exists():
            return pth
    p = row.get("out_path")
    if isinstance(p, str) and p.strip():
        pth = Path(p)
        if not pth.is_absolute():
            pth = repo_root / pth
        if pth.exists():
            return pth
    return None


def image_size_str(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return f"{w}x{h}"
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Filter by aesthetic & relevance and export base set.")
    ap.add_argument("--repo-root", type=str, default=".", help="Repository root")
    ap.add_argument("--metadata", type=str, default="metadata/images.csv", help="Path to images.csv")
    ap.add_argument("--intermediate", type=str, default="metadata/intermediate.csv", help="Intermediate CSV path")
    ap.add_argument("--out-base-dir", type=str, default="data_clean/base", help="Destination directory for kept images")
    ap.add_argument("--out-base-csv", type=str, default="metadata/base.csv", help="Output CSV of selected rows")
    ap.add_argument("--top-n", type=int, default=16000, help="How many images to keep globally")
    ap.add_argument("--min-relevance", type=float, default=60.0, help="Hard gate on ae_relevance")
    ap.add_argument("--min-aesthetic", type=float, default=60.0, help="Hard gate on ae_aesthetic")
    ap.add_argument("--missing-log", type=str, default="metadata/ae_filter_missing.txt", help="Log for missing files")
    ap.add_argument("--dry-run", action="store_true", help="Skip copies / CSV writes")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = repo_root / args.metadata
    intermediate_csv = repo_root / args.intermediate
    out_base_dir = repo_root / args.out_base_dir
    out_base_csv = repo_root / args.out_base_csv
    missing_log = repo_root / args.missing_log

    out_base_dir.mkdir(parents=True, exist_ok=True)
    missing_log.parent.mkdir(parents=True, exist_ok=True)
    intermediate_csv.parent.mkdir(parents=True, exist_ok=True)
    out_base_csv.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        print(f"[ERROR] Metadata not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(metadata_csv)

    required_cols = [
        "image_id", "source_url", "phash64",
        "ae_relevance", "ae_aesthetic",
        "is_nsfw", "dupe_removed",
        "path_resized", "out_path",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in {metadata_csv}: {missing}", file=sys.stderr)
        sys.exit(1)

    def truthy(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

    df_int = df[(~df["is_nsfw"].map(truthy)) & (~df["dupe_removed"].map(truthy))].copy()
    tmp = intermediate_csv.with_suffix(".csv.tmp")
    df_int.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(intermediate_csv)
    print(f"[INFO] Intermediate (no NSFW, no dupes): {intermediate_csv}  rows={len(df_int)}")

    gated = df_int[(df_int["ae_relevance"] >= args.min_relevance) &
                   (df_int["ae_aesthetic"] >= args.min_aesthetic)].copy()
    print(f"[INFO] After hard gate (rel≥{args.min_relevance}, aes≥{args.min_aesthetic}): rows={len(gated)}")
    if len(gated) == 0:
        print("[WARN] No rows passed hard gate. Exiting.")
        sys.exit(0)

    factor = 0.85 + 0.15 * (gated["ae_relevance"] - 60.0) / 40.0
    factor = factor.clip(lower=0.85, upper=1.0)
    gated["__effective"] = gated["ae_aesthetic"] * factor

    top = gated.sort_values("__effective", ascending=False).head(args.top_n).copy()
    print(f"[INFO] Selected top {len(top)} rows globally.")

    missing_paths: List[str] = []
    base_rows = []

    if args.dry_run:
        print("[DRY-RUN] Skipping file copies and base.csv writing.")
    else:
        top = top.reset_index(drop=True)
        for i in tqdm(range(len(top)), desc="Copying to data_clean/base"):
            row = top.iloc[i]
            src = resolve_resized_path(row, repo_root)
            if src is None:
                missing_paths.append(row.get("path_resized") or row.get("out_path") or "")
                continue

            dst = out_base_dir / Path(src).name
            try:
                shutil.copy2(src, dst)
            except Exception:
                missing_paths.append(str(src))
                continue

            base_rows.append({
                "path": str(Path(args.out_base_dir) / dst.name).replace("\\", "/"),
                "source_url": row.get("source_url", ""),
                "image_size": image_size_str(dst) or "",
                "ae_relevance": row.get("ae_relevance", ""),
                "ae_aesthetic": row.get("ae_aesthetic", ""),
                "image_id": row.get("image_id", ""),
                "phash": row.get("phash64", ""),
            })

        with open(missing_log, "w", encoding="utf-8") as f:
            if missing_paths:
                f.write("# Missing or failed to copy:\n")
                for p in missing_paths:
                    f.write(str(p) + "\n")

        base_cols = ["path", "source_url", "image_size", "ae_relevance", "ae_aesthetic", "image_id", "phash"]
        base_df = pd.DataFrame(base_rows, columns=base_cols)
        tmp_csv = out_base_csv.with_suffix(".csv.tmp")
        base_df.to_csv(tmp_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        tmp_csv.replace(out_base_csv)

    if not args.dry_run:
        kept = len([p for p in out_base_dir.iterdir() if p.is_file()])
        print("\n[SUMMARY]")
        print(f"  Requested top-N: {args.top_n}")
        print(f"  Copied to: {out_base_dir}")
        print(f"  Present in data_clean/base: {kept}")
        print(f"  Missing or failed copies: {len(missing_paths)} (log: {missing_log})" if missing_paths else "  Missing or failed copies: 0")
        print(f"  Wrote: {out_base_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()

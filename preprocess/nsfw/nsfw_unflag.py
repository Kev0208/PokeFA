"""
Manually unflag images by MD5 (remove NSFW drops; optional images.csv clear).

Usage:
  # Preview removals only
  python preprocess/nsfw/nsfw_unflag.py --root . --hash 9f6a12b6a9c3... h2.png --dry-run

  # Apply and also clear images.csv NSFW fields
  python preprocess/nsfw/nsfw_unflag.py --root . --hash 9f6a12b6a9c3... h2 --also-clear-images-ledger
"""

from __future__ import annotations
import argparse, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set
import pandas as pd

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def extract_md5s(tokens: Iterable[str]) -> List[str]:
    out: Set[str] = set()
    pat = re.compile(r"([a-fA-F0-9]{32})")
    for t in tokens:
        m = pat.search(t)
        if m: out.add(m.group(1).lower())
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(description="Remove NSFW drop rows by MD5; optional images.csv clear.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hash", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--also-clear-images-ledger", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    drops_path = root / "metadata" / "drops.csv"
    if not drops_path.exists():
        raise FileNotFoundError(f"Not found: {drops_path}")

    hashes = extract_md5s(args.hash)
    if not hashes: raise SystemExit("No valid 32-hex MD5s extracted.")

    df = pd.read_csv(drops_path)
    if df.empty:
        print("drops.csv empty; nothing to do.")
        return
    for c in ["stage","image_id","path_raw"]:
        if c not in df.columns: df[c] = ""

    stage = df["stage"].astype(str).str.lower() == "nsfw"
    id_lc = df["image_id"].astype(str).str.lower()
    path_lc = df["path_raw"].astype(str).str.lower()

    mask = pd.Series(False, index=df.index)
    for h in hashes:
        mask |= stage & ((id_lc == h) | (path_lc.str.contains(h)))

    n = int(mask.sum())
    if n == 0:
        print(f"No NSFW rows match {len(hashes)} hash(es).")
        return

    preview_cols = [c for c in ["ts","image_id","stage","drop_reason","band","species","path_raw"] if c in df.columns]
    print(f"Matches to remove: {n}")
    print(df.loc[mask, preview_cols].head(20).to_string(index=False))

    if args.dry_run:
        print("[DRY-RUN] No changes written.")
        return

    bak = drops_path.with_suffix(f".bak.{ts()}.csv")
    drops_path.replace(bak)
    df.loc[~mask].to_csv(drops_path, index=False)
    print(f"Removed {n} row(s). Backup: {bak}")
    print(f"Updated: {drops_path}")

    if args.also_clear_images_ledger:
        images_csv = root / "metadata" / "images.csv"
        if images_csv.exists():
            di = pd.read_csv(images_csv)
            for col in ["image_id","path_raw"]:
                if col not in di.columns: di[col] = ""
            id_i = di["image_id"].astype(str).str.lower()
            path_i = di["path_raw"].astype(str).str.lower()
            mask_i = pd.Series(False, index=di.index)
            for h in hashes:
                mask_i |= (id_i == h) | (path_i.str.contains(h))
            if mask_i.any():
                for c in ["NSFW","NSFW_tags","NSFW_score","NSFW_threshold","NSFW_model","NSFW_ts","NSFW_run_id"]:
                    if c not in di.columns: di[c] = ""
                di.loc[mask_i, "NSFW"] = 0
                di.loc[mask_i, ["NSFW_tags","NSFW_score","NSFW_threshold","NSFW_model","NSFW_ts","NSFW_run_id"]] = ""
                tmp = images_csv.with_suffix(".tmp.csv")
                di.to_csv(tmp, index=False); tmp.replace(images_csv)
                print(f"Cleared images.csv for {int(mask_i.sum())} image(s).")
            else:
                print("No images.csv rows matched.")
        else:
            print(f"Missing {images_csv}; skipped clear.")

if __name__ == "__main__":
    main()

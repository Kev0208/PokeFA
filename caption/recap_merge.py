"""
Merge re-captioned rows back into metadata/base.csv (by image_id or path).

Usage:
  python caption/recap_merge.py --base metadata/base.csv --caps metadata/base_capscreening.csv --inplace --backup
  python caption/recap_merge.py --base metadata/base.csv --caps metadata/base_capscreening.csv --only-missing
"""

import argparse, os, sys
import pandas as pd
from typing import Tuple

def read_csv_str(p): return pd.read_csv(p, dtype=str, keep_default_na=False)

def choose_key(base, caps, prefer: str) -> Tuple[str,str]:
    if prefer == "image_id" and "image_id" in base.columns and "image_id" in caps.columns: return "image_id",""
    if prefer == "path" and "path" in base.columns and "path" in caps.columns: return "path",""
    if "image_id" in base.columns and "image_id" in caps.columns: return "image_id",""
    if "path" in base.columns and "path" in caps.columns: return "path",""
    return "","no common key"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="metadata/base.csv")
    ap.add_argument("--caps", default="metadata/base_capscreening.csv")
    ap.add_argument("--key", choices=["auto","image_id","path"], default="auto")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--require-nonempty", action="store_true")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.base) or not os.path.exists(args.caps):
        print("[ERROR] input CSV(s) not found", file=sys.stderr); sys.exit(2)

    base, caps = read_csv_str(args.base), read_csv_str(args.caps)
    for c in ("caption","tags"):
        if c not in base.columns: base[c] = ""
        if c not in caps.columns: caps[c] = ""

    key,_ = choose_key(base, caps, args.key)
    if not key: print("[ERROR] cannot determine join key", file=sys.stderr); sys.exit(2)

    use = caps[[key,"caption","tags"]].copy()
    if args.require_nonempty:
        use = use[(use["caption"].str.strip()!="") & (use["tags"].str.strip()!="")]

    merged = base.merge(use, on=key, how="left", suffixes=("","_new"))
    take_cap = (merged["caption_new"].astype(str).str.strip()!="")
    take_tags= (merged["tags_new"].astype(str).str.strip()!="")
    if args.only-missing:
        take_cap &= (merged["caption"].astype(str).str.strip()=="")
        take_tags&= (merged["tags"].astype(str).str.strip()=="")

    merged.loc[take_cap, "caption"] = merged.loc[take_cap, "caption_new"]
    merged.loc[take_tags,"tags"] = merged.loc[take_tags, "tags_new"]
    merged.drop(columns=[c for c in ("caption_new","tags_new") if c in merged.columns], inplace=True)

    out_path = args.base if args.inplace else (args.out or "metadata/base_merged.csv")
    if args.inplace and args.backup:
        base.to_csv(args.base+".bak", index=False); print(f"[BACKUP] {args.base}.bak")
    merged.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path}  (updated_caption={int(take_cap.sum())} updated_tags={int(take_tags.sum())})")

if __name__ == "__main__":
    main()

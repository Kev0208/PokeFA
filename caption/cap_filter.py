"""
Screen captions that fail Pokémon identification; optionally emit recap CSV or prune.

Usage:
  # Screening report (default)
  python caption/cap_filter.py --repo-root /path --metadata metadata/base.csv

  # Clean re-caption file (keeps minimal cols; blanks caption/tags)
  python caption/cap_filter.py --repo-root /path --metadata metadata/base.csv --recap-clean

  # Prune in place and mark images.csv
  python caption/cap_filter.py --repo-root /path --metadata metadata/base.csv \
    --prune-inplace --backup --images metadata/images.csv --images-drop-col drop_unrecognized_pokemon
"""

import argparse, json, os, sys, shutil, tempfile
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Screen bad captions and (optionally) create a recap CSV or prune in place.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--tags-col", default="tags")
    p.add_argument("--pokemon-key", default="Pokémon")
    p.add_argument("--recap-clean", action="store_true")
    p.add_argument("--keep-originals", action="store_true")
    p.add_argument("--recap-columns", default="path,image_id,caption,tags")
    p.add_argument("--prune-inplace", action="store_true")
    p.add_argument("--backup", action="store_true")
    p.add_argument("--images", default=None)
    p.add_argument("--images-drop-col", default="drop_unrecognized_pokemon")
    return p.parse_args()

def _safe_json_loads(s):
    if pd.isna(s) or s is None or s == "" or not isinstance(s, str): return None
    try: return json.loads(s)
    except Exception:
        try: return json.loads(s.replace('""','"').strip().rstrip(","))
        except Exception: return None

def _atomic_write_csv(df, path, backup: bool):
    if backup and os.path.exists(path): shutil.copyfile(path, path+".bak")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, newline="", suffix=".tmp",
                                         dir=os.path.dirname(path) or ".") as tf:
            df.to_csv(tf.name, index=False); tmp=tf.name
        shutil.move(tmp, path)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

def main():
    args = parse_args()
    in_path = os.path.join(args.repo_root, args.metadata) if not os.path.isabs(args.metadata) else args.metadata
    out_path = args.out or os.path.join(os.path.dirname(in_path), "base_capscreening.csv")
    if not os.path.exists(in_path): print(f"[ERROR] {in_path} not found", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    if args.tags_col not in df.columns: print(f"[ERROR] '{args.tags_col}' missing", file=sys.stderr); sys.exit(2)

    missed_mask = []
    for s in df[args.tags_col].tolist():
        obj = _safe_json_loads(s)
        names = []
        if isinstance(obj, dict):
            val = obj.get(args.pokemon_key)
            if isinstance(val, list): names = [x.strip() for x in val if isinstance(x, str) and x.strip()]
            elif isinstance(val, str) and val.strip(): names = [val.strip()]
        missed_mask.append(len(names) == 0)

    out_df = df.loc[missed_mask].copy()

    if args.prune_inplace:
        kept_df = df.loc[[not m for m in missed_mask]].copy()
        _atomic_write_csv(kept_df, in_path, backup=args.backup)
        if args.images and os.path.exists(os.path.join(args.repo_root, args.images)):
            images_path = os.path.join(args.repo_root, args.images) if not os.path.isabs(args.images) else args.images
            images_df = pd.read_csv(images_path, dtype=str, keep_default_na=False)
            if "image_id" in images_df.columns and "image_id" in out_df.columns:
                col = args.images_drop_col
                if col not in images_df.columns: images_df[col] = ""
                mask = images_df["image_id"].astype(str).isin(out_df["image_id"].astype(str))
                if mask.any():
                    images_df.loc[mask, col] = "TRUE"
                    _atomic_write_csv(images_df, images_path, backup=args.backup)
        print(f"[PRUNE] removed={len(out_df)} kept={len(kept_df)} -> {in_path}")
        return

    if args.recap_clean:
        keep_cols = [c.strip() for c in args.recap_columns.split(",") if c.strip()]
        for c in keep_cols:
            if c not in out_df.columns: out_df[c] = ""
        if "caption" in out_df.columns: out_df["caption"] = ""
        if "tags" in out_df.columns: out_df["tags"] = ""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path}  (flagged={len(out_df)} of {len(df)})")

if __name__ == "__main__":
    main()

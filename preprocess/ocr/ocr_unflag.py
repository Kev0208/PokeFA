"""
Clear OCR flags for given MD5s in metadata CSV.

Usage:
  # Unflag by one or more MD5s (exact 32-hex strings)
  python preprocess/ocr_unflag.py --csv metadata/images.csv --id 9f6a12... a1b2c3...

  # From a file (one MD5 per line)
  python preprocess/ocr_unflag.py --csv metadata/images.csv --id-file md5_list.txt
"""

import argparse, os, tempfile, pandas as pd

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["has_ocr","ocr_mask_path","ocr_boxes_path"]:
        if c not in df.columns: df[c] = ""
    return df

def atomic_write_csv(df: pd.DataFrame, path: str):
    d = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".tmp") as tmp:
        df.to_csv(tmp.name, index=False); tmp_path = tmp.name
    os.replace(tmp_path, path)

def main():
    ap = argparse.ArgumentParser(description="Unset has_ocr and clear mask/boxes paths for provided MD5s.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id", dest="ids", action="append", default=[])
    ap.add_argument("--id-file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ids = list(args.ids)
    if args.id_file:
        with open(args.id_file, "r", encoding="utf-8") as f:
            ids += [ln.strip() for ln in f if ln.strip()]
    ids = sorted(set(ids))
    if not ids:
        raise SystemExit("Provide at least one --id or --id-file")

    df = pd.read_csv(args.csv)
    df = ensure_cols(df)

    if "image_id" not in df.columns:
        raise SystemExit("metadata/images.csv needs an image_id column")

    mask = df["image_id"].astype(str).str.lower().isin([h.lower() for h in ids])
    n = int(mask.sum())
    if n == 0:
        print("No rows match."); return

    df.loc[mask, "has_ocr"] = "False"
    df.loc[mask, ["ocr_mask_path","ocr_boxes_path"]] = ""

    print(f"Rows updated: {n}")
    if args.dry_run:
        print("[DRY-RUN] Skipped writing."); return
    atomic_write_csv(df, args.csv)
    print("Done.")

if __name__ == "__main__":
    main()

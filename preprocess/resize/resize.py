"""
Resize images per SDXL-friendly policy with deterministic kernels.

Usage
--------
  # Basic run (repo root holds data_raw/, metadata/, etc.)
  python preprocess/resize/resize.py --metadata metadata/images.csv --repo-root .

  # Custom inpaint and output roots
  python preprocess/resize/resize.py --metadata metadata/images.csv --repo-root . \
    --inpaint-root data_stage/OCR_inpainted --out-root data_stage/resized

  # Write a CSV backup before updating
  python preprocess/resize/resize.py --metadata metadata/images.csv --repo-root . --backup-csv

Policy
------
Size
  - min_side < 1024                → upscale to 1024 (Bicubic)
  - 1024 ≤ min_side ≤ 2048         → keep
  - min_side > 2048 or long > 4096 → downscale to min_side = 2048
      * if scale < 0.5 → AREA
      * else           → Bicubic

AR
  - If AR > 2.0 → no pre-crop; rely on center-biased random crops at train time
"""

import argparse
import logging
import os
import sys
import shutil
import pathlib

import pandas as pd
from PIL import Image, ImageOps

import cv2
import numpy as np

CSV_COLS = [
    "ts","scrape_run_id","image_id","species","rank","band","query",
    "source_url","source_domain","path_raw","fmt","w","h","content_type",
    "safe","hl","gl","is_nsfw","has_ocr","ocr_mask_path","ocr_boxes_path"
]

NEW_COLS = [
    "orig_w","orig_h","new_w","new_h","scale","kernel","ar_before","ar_after",
    "resize_action","source_used","status","status_msg","out_path"
]

TARGET_MIN = 1024
CAP_MIN    = 2048
CAP_LONG   = 4096
DEF_JPEG_Q = 95

DEFAULT_INPAINT_ROOT = "data_stage/OCR_inpainted"
DEFAULT_OUT_ROOT     = "data_stage/resized"

IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cv_area_resize(img: Image.Image, new_w: int, new_h: int) -> Image.Image:
    arr = np.array(img.convert("RGBA" if img.mode == "RGBA" else "RGB"))
    if arr.shape[-1] == 4:
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        a   = arr[:, :, 3]
        rgb_r = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        a_r   = cv2.resize(a,   (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgba  = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2RGB)
        out   = np.dstack([rgba, a_r])
        return Image.fromarray(out, mode="RGBA")
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr_r = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_r)


def pick_kernel(old_short: int, new_short: int, is_downscale: bool) -> str:
    if not is_downscale:
        return "BICUBIC"
    s = float(new_short) / float(old_short)
    return "AREA" if s < 0.5 else "BICUBIC"


def compute_target_size(w: int, h: int):
    short, long_ = (h, w) if h < w else (w, h)
    if short < TARGET_MIN:
        scale = TARGET_MIN / float(short)
        return int(round(w*scale)), int(round(h*scale)), scale, "BICUBIC", "upscale_to_1024"
    if short > CAP_MIN or long_ > CAP_LONG:
        scale = CAP_MIN / float(short)
        kernel_tag = pick_kernel(short, CAP_MIN, is_downscale=True)
        return int(round(w*scale)), int(round(h*scale)), scale, kernel_tag, "downscale_to_2048"
    return w, h, 1.0, "KEEP", "keep"


def save_image(img: Image.Image, dst_path: str, jpeg_q: int = DEF_JPEG_Q):
    ensure_dir(dst_path)
    ext = pathlib.Path(dst_path).suffix.lower()
    if ext == ".png":
        img.save(dst_path, format="PNG", optimize=True)
    elif ext == ".webp":
        img.save(dst_path, format="WEBP", quality=90, method=6)
    elif ext in {".jpg", ".jpeg"}:
        if img.mode in {"RGBA","P"}:
            img = img.convert("RGB")
        img.save(dst_path, format="JPEG", quality=jpeg_q, optimize=True, subsampling=0)
    else:
        if img.mode in {"RGBA","P"}:
            img.save(dst_path, format="PNG", optimize=True)
        else:
            img = img.convert("RGB")
            img.save(dst_path, format="JPEG", quality=jpeg_q, optimize=True, subsampling=0)


def resolve_repo_path(repo_root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))


def inpaint_png_path(inpaint_root: str, band: str, species: str, image_id: str) -> str:
    return os.path.join(inpaint_root, band, species, image_id + ".png")


def process_one_row(row: pd.Series, repo_root: str, inpaint_root: str, out_root: str, jpeg_q: int):
    image_id = str(row["image_id"]).strip()
    band     = str(row["band"]).strip()
    species  = str(row["species"]).strip()

    if str(row["is_nsfw"]).strip().lower() in {"true","1","yes","y","t"}:
        return {"status":"skipped_nsfw","status_msg":"","source_used":None,"resize_action":None,"kernel":None,
                "orig_w":None,"orig_h":None,"new_w":None,"new_h":None,"scale":None,"ar_before":None,"ar_after":None,
                "out_path":None}

    path_raw_csv = (row.get("path_raw") or "").strip()
    path_raw = resolve_repo_path(repo_root, path_raw_csv) if path_raw_csv else ""

    has_ocr = str(row["has_ocr"]).strip().lower() in {"true","1","yes","y","t"}
    inpaint_path = inpaint_png_path(inpaint_root, band, species, image_id) if has_ocr else None

    source_used = source_path = None
    if has_ocr and inpaint_path and os.path.isfile(inpaint_path):
        source_used, source_path = "inpainted", inpaint_path
    elif path_raw and os.path.isfile(path_raw):
        source_used, source_path = "raw", path_raw
    else:
        return {"status":"missing","status_msg":"source not found","source_used":None,"resize_action":None,"kernel":None,
                "orig_w":None,"orig_h":None,"new_w":None,"new_h":None,"scale":None,"ar_before":None,"ar_after":None,
                "out_path":None}

    try:
        img = Image.open(source_path)
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        return {"status":"error_open","status_msg":f"{e}","source_used":source_used,"resize_action":None,"kernel":None,
                "orig_w":None,"orig_h":None,"new_w":None,"new_h":None,"scale":None,"ar_before":None,"ar_after":None,
                "out_path":None}

    w, h = img.size
    short, long_ = (h, w) if h < w else (w, h)
    ar_before = (long_ / short) if short > 0 else 0.0

    new_w, new_h, scale, kernel_tag, action = compute_target_size(w, h)

    if action == "keep":
        out_img = img
        kernel_used = "KEEP"
    else:
        if kernel_tag == "AREA":
            out_img = cv_area_resize(img, new_w, new_h)
            kernel_used = "AREA"
        else:
            out_img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            kernel_used = "BICUBIC"

    new_w2, new_h2 = out_img.size
    short2, long2 = (new_h2, new_w2) if new_h2 < new_w2 else (new_w2, new_h2)
    ar_after = (long2 / short2) if short2 > 0 else 0.0

    if source_used == "inpainted":
        out_ext = ".png"
    else:
        sfx = pathlib.Path(source_path).suffix.lower()
        out_ext = sfx if sfx in IMG_EXTS else (".png" if out_img.mode in {"RGBA","P"} else ".jpg")

    out_path = os.path.join(out_root, band, species, image_id + out_ext)
    try:
        save_image(out_img, out_path, jpeg_q=DEF_JPEG_Q)
    except Exception as e:
        return {"status":"error_save","status_msg":f"{e}","source_used":source_used,"resize_action":action,"kernel":kernel_used,
                "orig_w":w,"orig_h":h,"new_w":None,"new_h":None,"scale":round(scale,6),"ar_before":round(ar_before,6),
                "ar_after":None,"out_path":None}

    return {"status":"ok","status_msg":"","source_used":source_used,"resize_action":action,"kernel":kernel_used,
            "orig_w":w,"orig_h":h,"new_w":new_w2,"new_h":new_h2,"scale":round(scale,6),
            "ar_before":round(ar_before,6),"ar_after":round(ar_after,6),"out_path":out_path}


def main():
    ap = argparse.ArgumentParser(description="Resize images per policy; prefer OCR-inpainted PNGs; no raw deletions.")
    ap.add_argument("--metadata", required=True, help="Path to metadata/images.csv")
    ap.add_argument("--repo-root", required=True, help="Repo root that contains data_raw/ etc.")
    ap.add_argument("--inpaint-root", default=DEFAULT_INPAINT_ROOT, help="Root for OCR-inpainted PNGs")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Destination root for resized images")
    ap.add_argument("--backup-csv", action="store_true", help="Write a .bak of the CSV before updating")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    repo_root = os.path.abspath(args.repo_root)
    inpaint_root = args.inpaint_root if os.path.isabs(args.inpaint_root) else os.path.join(repo_root, args.inpaint_root)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(repo_root, args.out_root)

    csv_path = args.metadata if os.path.isabs(args.metadata) else os.path.join(repo_root, args.metadata)
    if not os.path.isfile(csv_path):
        logging.error("CSV not found: %s", csv_path); sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_COLS if c not in df.columns]
    if missing:
        logging.error("CSV missing required columns: %s", missing); sys.exit(1)

    if args.backup_csv:
        shutil.copy2(csv_path, csv_path + ".bak")
        logging.info("Backed up CSV to %s.bak", csv_path)

    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = None

    total = len(df)
    ok = skipped = errors = 0

    for i, row in df.iterrows():
        res = process_one_row(row, repo_root, inpaint_root, out_root, DEF_JPEG_Q)
        for k, v in res.items():
            if k not in df.columns:
                df[k] = None
            df.at[i, k] = v

        st = res.get("status", "")
        if st == "ok":
            ok += 1
        elif st.startswith("skipped"):
            skipped += 1
        elif st.startswith("error") or st == "missing":
            errors += 1
        else:
            skipped += 1

        if (i + 1) % 200 == 0:
            logging.info("Progress: %d/%d (ok=%d, skipped=%d, errors=%d)", i+1, total, ok, skipped, errors)

    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

    logging.info("Done. Processed %d rows: ok=%d, skipped=%d, errors=%d", total, ok, skipped, errors)
    logging.info("Output root: %s", out_root)


if __name__ == "__main__":
    main()

"""
Flag NSFW images using DeepDanbooru (v3) tags.

Usage:
  python preprocess/nsfw/nsfw_filter.py --root . --dd-project /path/to/deepdanbooru-project --threshold 0.5
  python preprocess/nsfw/nsfw_filter.py --root . --dd-project /path/to/project --tags nude,sex,breasts --threshold 0.4
"""

from __future__ import annotations
import argparse, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

DEFAULT_TAGS = ["nude","sex","pussy","penis","breasts","nipples","cum","vagina"]

@dataclass
class NSFWConfig:
    tags: List[str]
    threshold: float
    min_side: int

def ts_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def run_id() -> str:
    return "nsfw_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def ensure_dirs(root: Path) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "data_stage" / "nsfw").mkdir(parents=True, exist_ok=True)

def read_images_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    for c in ["NSFW","NSFW_tags","NSFW_score","NSFW_threshold","NSFW_model","NSFW_ts","NSFW_run_id"]:
        if c not in df.columns: df[c] = ""
    return df

def write_images_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def append_drop(drops_path: Path, row: Dict) -> None:
    new = not drops_path.exists()
    with drops_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","image_id","url","stage","drop_reason","nsfw_tags","nsfw_score","threshold","band","species","path_raw"
        ])
        if new: w.writeheader()
        w.writerow(row)

def band_from(path_raw: Path) -> str:
    try: return path_raw.parents[2].name
    except Exception: return ""

def species_folder_from(path_raw: Path) -> str:
    try: return path_raw.parents[1].name
    except Exception: return ""

def load_rgb(path: Path, min_side: int) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            im.load()
            if min(im.size) < min_side: return None
            return im.convert("RGB") if im.mode != "RGB" else im
    except Exception:
        return None

def score_nsfw(preds: Dict[str,float], tags: List[str]) -> Tuple[float, List[Tuple[str,float]]]:
    hits = [(t, float(preds.get(t, 0.0))) for t in tags]
    hits = [h for h in hits if h[1] > 0.0]
    hits.sort(key=lambda x: x[1], reverse=True)
    return (hits[0][1] if hits else 0.0, hits)

def load_dd_commands(project_dir: Path):
    from deepdanbooru.commands import load_model_from_project, load_tags_from_project, tag_image
    tags = load_tags_from_project(project_dir)
    model = load_model_from_project(project_dir, compile_model=False)
    return model, tags, tag_image

def tag_with_dd(im: Image.Image, model, tags, tag_image_fn) -> Dict[str,float]:
    arr = np.asarray(im)
    y = tag_image_fn(image=arr, model=model, tags=tags, use_gpu=False, threshold=0.0, conservative=False)
    return {str(k).lower(): float(v) for k, v in y.items()}

def main():
    ap = argparse.ArgumentParser(description="DeepDanbooru-based NSFW flagging; non-destructive.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--dd-project", required=True, help="Folder with DeepDanbooru project.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS))
    ap.add_argument("--min-side", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    ensure_dirs(root)

    images_csv = root / "metadata" / "images.csv"
    drops_csv  = root / "metadata" / "drops.csv"
    df = read_images_csv(images_csv)

    sexual_tags = [t.strip().lower().replace(" ","_") for t in args.tags.split(",") if t.strip()]
    cfg = NSFWConfig(tags=sexual_tags, threshold=args.threshold, min_side=args.min_side)

    project_dir = Path(args.dd_project)
    if not (project_dir / "project.json").exists():
        raise FileNotFoundError(f"project.json not found under {project_dir}")

    model, tags, tag_image_fn = load_dd_commands(project_dir)
    model_label = f"deepdanbooru:{project_dir.name} (commands)"
    now, rid = ts_iso(), run_id()

    flagged = 0
    it = df.itertuples(index=False)
    for row in tqdm(it, total=len(df), desc="NSFW scan"):
        path_raw = Path(getattr(row, "path_raw", ""))
        if not path_raw.exists(): continue
        im = load_rgb(path_raw, cfg.min_side)
        if im is None: continue

        preds = tag_with_dd(im, model, tags, tag_image_fn)
        score, hits = score_nsfw(preds, cfg.tags)
        is_nsfw = int(score >= cfg.threshold)
        hit_tags = ";".join([f"{t}:{p:.3f}" for t, p in hits if p >= 0.01])

        image_id = getattr(row, "image_id", "")
        mask = (df["image_id"] == image_id) if image_id else (df["path_raw"] == str(path_raw))

        df.loc[mask, "NSFW"] = is_nsfw
        df.loc[mask, "NSFW_tags"] = hit_tags
        df.loc[mask, "NSFW_score"] = f"{score:.3f}"
        df.loc[mask, "NSFW_threshold"] = f"{cfg.threshold:.3f}"
        df.loc[mask, "NSFW_model"] = model_label
        df.loc[mask, "NSFW_ts"] = now
        df.loc[mask, "NSFW_run_id"] = rid

        if is_nsfw:
            band = band_from(path_raw)
            species_folder = species_folder_from(path_raw)
            dest = root / "data_stage" / "nsfw" / band / species_folder
            dest.mkdir(parents=True, exist_ok=True)
            try:
                (dest / path_raw.name).write_bytes(path_raw.read_bytes())
            except Exception:
                pass
            append_drop(drops_csv, {
                "ts": now,
                "image_id": image_id or "",
                "url": getattr(row, "source_url", ""),
                "stage": "nsfw",
                "drop_reason": "nsfw",
                "nsfw_tags": hit_tags,
                "nsfw_score": f"{score:.3f}",
                "threshold": f"{cfg.threshold:.3f}",
                "band": band,
                "species": getattr(row, "species", species_folder),
                "path_raw": str(path_raw),
            })
            flagged += 1

    write_images_csv(images_csv, df)
    print(f"[NSFW] flagged={flagged} threshold={cfg.threshold} tags={','.join(cfg.tags)} model={model_label}")

if __name__ == "__main__":
    main()
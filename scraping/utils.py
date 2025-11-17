from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timezone

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

BAND_NAMES = {"band1", "band2", "band3", "band4"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def make_scrape_run_id() -> str:
    return "scr_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def safe_species_folder_name(rank: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "", name.strip().replace(" ", "_"))
    return f"{rank:04d}_{safe}"

def band_label_from_csv(band_field: str) -> str:
    s = str(band_field).strip()
    if s in BAND_NAMES:
        return s
    if s.startswith("1-150"): return "band1"
    if s.startswith("151-500"): return "band2"
    if s.startswith("501-800"): return "band3"
    if s.startswith("801-1025"): return "band4"
    raise ValueError(f"Unrecognized band value: {band_field}")

def ensure_root_skeleton(root: Path) -> None:
    for d in [
        "data_raw/band1","data_raw/band2","data_raw/band3","data_raw/band4",
        "data_cache/serpapi_queries","data_cache/http","logs","state","metadata"
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

def ensure_species_dirs(root: Path, band: str, species_folder: str) -> Tuple[Path, Path]:
    species_dir = root / "data_raw" / band / species_folder
    imgs_dir = species_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return species_dir, imgs_dir

def urls_jsonl_path(root: Path, band: str, species_folder: str) -> Path:
    return root / "data_raw" / band / species_folder / "urls.jsonl"

def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.setdefault("ts", iso_now())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

IMAGES_CSV_HEADERS = [
    "ts","scrape_run_id",
    "image_id","species","rank","band",
    "query","source_url","source_domain",
    "path_raw","fmt","w","h","content_type",
    "safe","hl","gl"
]

def images_csv_path(root: Path) -> Path:
    return root / "metadata" / "images.csv"

def ensure_images_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(IMAGES_CSV_HEADERS)

def append_image_row(path: Path, row: Dict) -> None:
    ensure_images_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, "") for h in IMAGES_CSV_HEADERS])

def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def choose_ext_from_url(url: str) -> str:
    base = url.split("?")[0]
    _, ext = os.path.splitext(base)
    ext = ext.lower()
    return ext if ext in {".jpg",".jpeg",".png",".webp"} else ".jpg"

def choose_ext_from_url_or_fmt(url: str, fmt: Optional[str]) -> str:
    if fmt == "JPEG": return ".jpg"
    if fmt == "PNG":  return ".png"
    if fmt == "WEBP": return ".webp"
    return choose_ext_from_url(url)

def download_image(url: str, timeout: int = 15) -> Optional[Tuple[bytes, str]]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if not ctype.startswith("image/"):
            return None
        return resp.content, ctype
    except Exception:
        return None

def validate_image_bytes(blob: bytes, min_side: int = 128):
    try:
        bio = io.BytesIO(blob)
        with Image.open(bio) as im:
            im.verify()
        bio2 = io.BytesIO(blob)
        with Image.open(bio2) as im:
            im.load()
            fmt = (im.format or "").upper()
            w, h = im.size
        if fmt not in ALLOWED_FORMATS or min(w, h) < min_side:
            return None
        return fmt, w, h
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    except Exception:
        return None

def load_quotas(quotas_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(quotas_csv)
    required = {"Ranking", "Name", "q_r_base", "band"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"quotas CSV missing columns: {missing}")
    df["band_label"] = df["band"].apply(band_label_from_csv)
    df["Ranking"] = df["Ranking"].astype(int)
    df["q_r_base"] = df["q_r_base"].astype(int)
    return df.sort_values(["band_label", "Ranking"]).reset_index(drop=True)

def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

class UrlSeen:
    def __init__(self):
        self._seen = set()
    def add(self, url: str) -> bool:
        if not url or url in self._seen:
            return False
        self._seen.add(url)
        return True

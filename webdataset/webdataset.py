"""
Pack train/val into WebDataset tar shards with hybrid captions and JSON metadata; emit manifests.

Usage:
  python webdataset/webdataset.py \
    --base metadata/base.csv \
    --out-root webdataset \
    --train-shards 16 \
    --overwrite
"""
import argparse
import io
import json
import math
import os
import random
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _read_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    for c in ("path", "image_id", "hybrid_caption", "split"):
        if c not in df.columns:
            raise SystemExit(f"[FATAL] base CSV missing column: {c}")
    return df


def _ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _parse_image_size(s: str):
    try:
        w, h = str(s).split("x", 1)
        w = int(w.strip()); h = int(h.strip())
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return None, None


def _parse_species_list(row: Dict[str, Any]) -> list:
    raw = str(row.get("species", "")).strip()
    if not raw:
        return []
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _load_image_bytes(p: Path) -> Optional[bytes]:
    try:
        raw = p.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        return raw
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardWriter:
    def __init__(self, pattern: str, maxcount: int):
        self.pattern = pattern
        self.maxcount = maxcount
        self.shard_idx = -1
        self.count = 0
        self.tar: Optional[tarfile.TarFile] = None

    def _next(self):
        if self.tar is not None:
            self.tar.close()
        self.shard_idx += 1
        self.count = 0
        outpath = self.pattern.format(self.shard_idx)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        self.tar = tarfile.open(outpath, "w")

    def write(self, key: str, files: Dict[str, bytes]):
        if self.tar is None or self.count >= self.maxcount:
            self._next()
        for suffix, data in files.items():
            info = tarfile.TarInfo(name=f"{key}{suffix}")
            info.size = len(data)
            info.mtime = 0
            self.tar.addfile(info, io.BytesIO(data))
        self.count += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def _build_counts(sub: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for _, r in sub.iterrows():
        for sp in set(_parse_species_list(r.to_dict())):
            counts[sp] = counts.get(sp, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["pokemon", "count", "fraction"])
    out = pd.DataFrame([{"pokemon": k, "count": v} for k, v in counts.items()]).sort_values(
        ["count", "pokemon"], ascending=[False, True]
    )
    out["fraction"] = out["count"] / out["count"].sum()
    return out


def _resolve(repo_root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (repo_root / p)


def _build_files(repo_root: Path, row: Dict[str, Any]) -> Optional[Dict[str, bytes]]:
    img_path = _resolve(repo_root, str(row["path"]))
    if not img_path.exists():
        return None
    img_bytes = _load_image_bytes(img_path)
    if img_bytes is None:
        return None

    cap = str(row.get("hybrid_caption", "")).strip().encode("utf-8")
    w, h = _parse_image_size(row.get("image_size", ""))

    meta = {
        "image_id": str(row.get("image_id", "")).strip(),
        "split": str(row.get("split", "")).strip(),
        "source_url": str(row.get("source_url", "")).strip(),
        "width": w,
        "height": h,
        "ae_relevance": float(row["ae_relevance"]) if str(row.get("ae_relevance", "")).strip() else None,
        "ae_aesthetic": float(row["ae_aesthetic"]) if str(row.get("ae_aesthetic", "")).strip() else None,
        "species": str(row.get("species", "")).strip(),
        "species_list": _parse_species_list(row),
        "tags": str(row.get("tags", "")).strip(),
    }
    if str(row.get("phash", "")).strip():
        meta["phash"] = str(row["phash"]).strip()

    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    ext = img_path.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        ext = ".jpg"

    return {ext: img_bytes, ".txt": cap, ".json": meta_bytes}


def build_shards(df: pd.DataFrame, repo_root: Path, out_root: Path, train_shards: int, seed: int, overwrite: bool):
    data_dir = out_root / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    mani_dir = out_root / "manifests"
    _ensure_dirs(train_dir, val_dir, mani_dir)

    if overwrite:
        for p in train_dir.glob("train-*.tar"):
            p.unlink(missing_ok=True)
        for p in val_dir.glob("val-*.tar"):
            p.unlink(missing_ok=True)

    n_total = len(df)
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())

    train_species = _build_counts(df[df["split"] == "train"])
    val_species = _build_counts(df[df["split"] == "val"])
    train_species.to_csv(mani_dir / "train_species.csv", index=False)
    val_species.to_csv(mani_dir / "val_species.csv", index=False)

    train_ids_path = mani_dir / "train_ids.txt"
    val_ids_path = mani_dir / "val_ids.txt"
    train_ids = sorted([str(x).strip() for x in df[df["split"] == "train"]["image_id"].tolist() if str(x).strip()])
    val_ids = sorted([str(x).strip() for x in df[df["split"] == "val"]["image_id"].tolist() if str(x).strip()])
    train_ids_path.write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    val_ids_path.write_text("\n".join(val_ids) + ("\n" if val_ids else ""))

    split_stats = {
        "total": n_total,
        "train": n_train,
        "val": n_val,
        "ids": {
            "train": {"path": "manifests/train_ids.txt", "count": len(train_ids), "sha256": _sha256(train_ids_path)},
            "val": {"path": "manifests/val_ids.txt", "count": len(val_ids), "sha256": _sha256(val_ids_path)},
        },
    }
    (mani_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2))

    random.seed(seed)
    train_rows = df[df["split"] == "train"].to_dict(orient="records")
    random.shuffle(train_rows)
    val_rows = df[df["split"] == "val"].to_dict(orient="records")

    train_n = len(train_rows)
    train_shard_size = max(1, math.ceil(train_n / max(1, train_shards)))
    train_writer = ShardWriter(str(train_dir / "train-{:05d}.tar"), maxcount=train_shard_size)
    val_writer = ShardWriter(str(val_dir / "val-{:05d}.tar"), maxcount=max(1, len(val_rows)))

    missing = 0
    seen = set()

    for row in tqdm(train_rows, desc="Packing train", unit="img"):
        key = str(row.get("image_id", "")).strip()
        if not key or key in seen:
            continue
        files = _build_files(repo_root, row)
        if not files:
            missing += 1
            continue
        train_writer.write(key, files)
        seen.add(key)

    for row in tqdm(val_rows, desc="Packing val", unit="img"):
        key = str(row.get("image_id", "")).strip()
        if not key or key in seen:
            continue
        files = _build_files(repo_root, row)
        if not files:
            missing += 1
            continue
        val_writer.write(key, files)
        seen.add(key)

    train_writer.close()
    val_writer.close()
    print(f"[DONE] train={len(train_ids)} val={len(val_ids)} missing_or_corrupt={missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="metadata/base.csv")
    ap.add_argument("--out-root", type=str, default="webdataset")
    ap.add_argument("--train-shards", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    base_csv = Path(args.base).resolve()
    out_root = Path(args.out_root).resolve()

    if not base_csv.exists():
        raise SystemExit(f"[FATAL] base CSV not found: {base_csv}")

    df = _read_base(base_csv)
    build_shards(df, repo_root, out_root, train_shards=args.train_shards, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

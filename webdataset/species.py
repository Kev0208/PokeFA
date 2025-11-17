"""
Normalize per-image species (form-agnostic), drop anomaly tokens, write global counts, and tag drops in images.csv.

Usage:
  python webdataset/species.py \
    --csv metadata/base.csv \
    --images metadata/images.csv \
    --out metadata/species.csv
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


MAP = {
    "tapufini": "tapu_fini",
    "tapulele": "tapu_lele",
    "coffagrigus": "cofagrigus",
    "giraffarig": "girafarig",
    "nidoran_f": "nidoran_female",
    "nidoran_m": "nidoran_male",
    "alolan_ninetales": "ninetales",
    "alolan_raichu": "raichu",
    "alolan_marowak": "marowak",
    "alolan_sandshrew": "sandshrew",
    "alolan_sandslash": "sandslash",
    "galarian_ponyta": "ponyta",
    "paldean_wooper": "wooper",
    "hisuian_arcanine": "arcanine",
    "urshifu_rapid_strike": "urshifu",
    "hoopa_unbound": "hoopa",
    "alolan_vulpix": "vulpix",
}

DROP = {
    "pokemon","original_pokemon","unknown","unknown_pokemon","unidentified_pokemon",
    "unknown_fakemon","unknown_pok_mon",
    "dragon_like_1","dragon_like_2","serpentine_pokemon","wild_duck",
    "fakemon","fan_made","fanmade_pokemon","fan_made_pokemon","original_fakemon",
    "original_fakemon_big","original_fakemon_small","fan_made_jelly_large",
    "fan_made_jelly_small","fan_monster_large","fan_monster_small",
    "fanmon_blue","fanmon_red",
    "ashmish","cinnapup","dread_dragoona","kamitsu_orochi",
    "fakemon_bloombird","fakemon_fire_crocodile","fakemon_gem_turtle","fakemon_sprout",
    "nidoran",
}


def _parse_tags_cell(s: str) -> dict:
    if not s or not isinstance(s, str):
        return {}
    try:
        return json.loads(s.strip())
    except Exception:
        return {}


def _pokemon_list(tags_dict: dict) -> list:
    if not isinstance(tags_dict, dict):
        return []
    pokes = tags_dict.get("Pokémon") or tags_dict.get("Pokemon")
    if not isinstance(pokes, list):
        return []
    out = []
    for name in pokes:
        if name is None:
            continue
        n = str(name).strip().lower()
        if n:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="metadata/species.csv")
    ap.add_argument("--min-count", type=int, default=0)
    args = ap.parse_args()

    base_csv = Path(args.csv)
    images_csv = Path(args.images)
    out_counts = Path(args.out)

    if not base_csv.exists():
        raise SystemExit(f"Input CSV not found: {base_csv}")
    if not images_csv.exists():
        raise SystemExit(f"Images CSV not found: {images_csv}")

    counts = Counter()
    dropped_ids = set()
    kept_rows = []
    per_image_species = {}

    with base_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "tags" not in reader.fieldnames or "image_id" not in reader.fieldnames:
            raise SystemExit("base.csv must contain 'image_id' and 'tags' columns.")
        fieldnames = list(reader.fieldnames)
        if "species" not in fieldnames:
            fieldnames.append("species")

        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue

            tags_dict = _parse_tags_cell(row.get("tags", ""))
            raw_species = _pokemon_list(tags_dict)

            if any(sp in DROP for sp in raw_species):
                dropped_ids.add(image_id)
                continue

            norm = []
            for sp in raw_species:
                sp = MAP.get(sp, sp)
                if sp in DROP:
                    continue
                norm.append(sp)

            seen = set()
            uniq = [s for s in norm if not (s in seen or seen.add(s))]

            per_image_species[image_id] = ", ".join(uniq)
            row["species"] = per_image_species[image_id]
            kept_rows.append(row)

            for sp in set(uniq):
                counts[sp] += 1

    out_counts.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(((sp, c) for sp, c in counts.items() if c >= args.min_count),
                   key=lambda x: (-x[1], x[0]))
    with out_counts.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pokemon", "count"])
        for sp, c in items:
            w.writerow([sp, c])

    base_backup = base_csv.with_suffix(base_csv.suffix + ".bak")
    shutil.copy2(base_csv, base_backup)
    tmp_path = base_csv.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)
    os.replace(tmp_path, base_csv)

    images_backup = images_csv.with_suffix(images_csv.suffix + ".bak")
    shutil.copy2(images_csv, images_backup)
    flag_col = "species_anomalies_drop"

    with images_csv.open(newline="", encoding="utf-8") as f_in, \
         images_csv.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fn = list(reader.fieldnames) if reader.fieldnames else []
        if "image_id" not in fn:
            raise SystemExit("images.csv must contain 'image_id' column.")
        if flag_col not in fn:
            fn.append(flag_col)

        writer = csv.DictWriter(f_out, fieldnames=fn)
        writer.writeheader()
        for row in reader:
            img_id = (row.get("image_id") or "").strip()
            row[flag_col] = "TRUE" if img_id in dropped_ids else (row.get(flag_col, "") or "")
            writer.writerow(row)
    os.replace(images_csv.with_suffix(".tmp"), images_csv)

    print(f"[WROTE] {out_counts} (species={len(items)})")
    print(f"[UPDATED] {base_csv} (backup={base_backup.name}; dropped={len(dropped_ids)} kept={len(kept_rows)})")
    print(f"[UPDATED] {images_csv} (backup={images_backup.name}; flagged={len(dropped_ids)})")


if __name__ == "__main__":
    main()

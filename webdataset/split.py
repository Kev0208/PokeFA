"""
Assign train/val splits with head-species quotas (random per-species sampling, no greedy cover).

Usage:
  python webdataset/split.py \
    --base metadata/base.csv \
    --species metadata/species.csv \
    --seed 42 \
    --val_ids manifests/val_ids.txt \
    --train_ids manifests/train_ids.txt
"""
import argparse
import os
import random
import shutil
from collections import Counter, defaultdict

import pandas as pd


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="metadata/base.csv")
    ap.add_argument("--species", type=str, default="metadata/species.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top2", type=int, default=100)
    ap.add_argument("--one_from", type=int, default=101)
    ap.add_argument("--one_to", type=int, default=301)
    ap.add_argument("--val_ids", type=str, default=None)
    ap.add_argument("--train_ids", type=str, default=None)
    return ap.parse_args()


def _norm_species_list(cell: str):
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = [p.strip().lower() for p in s.split(",")]
    return [p for p in parts if p]


def _count_by_species(df: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for sp_list in df["species_list"]:
        for sp in set(sp_list):
            counter[sp] += 1
    out = pd.DataFrame([{"pokemon": sp, "count": c} for sp, c in counter.items()])
    if len(out) == 0:
        return pd.DataFrame(columns=["pokemon", "count"])
    return out.sort_values(["count", "pokemon"], ascending=[False, True])


def main():
    args = _args()
    random.seed(args.seed)

    base_path = args.base
    species_path = args.species

    if not os.path.exists(base_path):
        raise SystemExit(f"base CSV not found: {base_path}")
    if not os.path.exists(species_path):
        raise SystemExit(f"species CSV not found: {species_path}")

    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    if "image_id" not in base.columns or "species" not in base.columns:
        raise SystemExit("metadata/base.csv must contain 'image_id' and 'species' columns.")

    sp_df = pd.read_csv(species_path, dtype={"pokemon": str, "count": int}, keep_default_na=False)
    if "pokemon" not in sp_df.columns or "count" not in sp_df.columns:
        raise SystemExit("metadata/species.csv must contain 'pokemon' and 'count' columns.")

    base["species_list"] = base["species"].apply(_norm_species_list)
    sp_df = sp_df.sort_values(["count", "pokemon"], ascending=[False, True]).reset_index(drop=True)
    sp_df["rank"] = sp_df.index + 1

    quotas = {}
    for _, r in sp_df.iterrows():
        rk = int(r["rank"])
        name = r["pokemon"].strip().lower()
        if rk <= args.top2:
            quotas[name] = 2
        elif args.one_from <= rk <= args.one_to:
            quotas[name] = 1

    cand_idx = defaultdict(list)
    for i, lst in enumerate(base["species_list"].tolist()):
        for sp in set(lst):
            cand_idx[sp].append(i)

    selected = set()
    for sp, need in quotas.items():
        pool = [i for i in cand_idx.get(sp, []) if i not in selected]
        if not pool:
            continue
        k = min(need, len(pool))
        for p in random.sample(pool, k):
            selected.add(p)

    base["split"] = "train"
    if selected:
        base.loc[list(selected), "split"] = "val"

    backup = base_path + ".bak"
    shutil.copyfile(base_path, backup)
    base.to_csv(base_path, index=False)
    print(f"[OK] Updated {base_path} (backup={backup})")
    print(f"[VAL] Selected {len(selected)} images.")

    if args.val_ids:
        os.makedirs(os.path.dirname(args.val_ids), exist_ok=True)
        base.loc[base["split"] == "val", "image_id"].to_csv(args.val_ids, index=False, header=False)
        print(f"[OK] Wrote {args.val_ids}")
    if args.train_ids:
        os.makedirs(os.path.dirname(args.train_ids), exist_ok=True)
        base.loc[base["split"] == "train", "image_id"].to_csv(args.train_ids, index=False, header=False)
        print(f"[OK] Wrote {args.train_ids}")

    train = base[base["split"] == "train"].copy()
    val = base[base["split"] == "val"].copy()

    def _add_fraction(df_counts: pd.DataFrame) -> pd.DataFrame:
        if len(df_counts) == 0:
            df_counts["fraction"] = []
            return df_counts
        total = df_counts["count"].sum()
        df_counts["fraction"] = df_counts["count"].astype(float) / float(total) if total else 0.0
        return df_counts

    train_counts = _add_fraction(_count_by_species(train))
    val_counts = _add_fraction(_count_by_species(val))

    train_counts.to_csv(os.path.join(os.path.dirname(species_path), "train_species.csv"), index=False)
    val_counts.to_csv(os.path.join(os.path.dirname(species_path), "val_species.csv"), index=False)
    print("[OK] Wrote train/val species summaries")


if __name__ == "__main__":
    main()

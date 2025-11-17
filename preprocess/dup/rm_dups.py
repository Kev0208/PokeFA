"""
Auto-mark near/identical duplicates across the whole dataset using 64-bit pHash (metadata-only).

Usage:
  python preprocess/dup/rm_dups.py \
    --metadata metadata/images.csv \
    --near-thresh 3 \
    --prefix-hex 4 \
    --audit-dir data_stage/dups \
    --backup-csv
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from imagehash import hex_to_hash


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    return str(x).strip()


def to_bool(x) -> bool:
    return s(x).lower() in {"true", "1", "yes", "y", "t"}


def hamming(ph1: str, ph2: str) -> int:
    return hex_to_hash(ph1) - hex_to_hash(ph2)


def f_or(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def i_or(x, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def resolution_score(row: pd.Series) -> int:
    nw, nh = i_or(row.get("new_w"), 0), i_or(row.get("new_h"), 0)
    if nw > 0 and nh > 0:
        return nw * nh
    w, h = i_or(row.get("w"), 0), i_or(row.get("h"), 0)
    return w * h


def choose_representative(idxs: List[int], df: pd.DataFrame) -> Tuple[int, str]:
    """
    Pick representative: highest ae_aesthetic -> highest resolution -> earliest ts.
    Returns (df_index, keep_reason).
    """
    cand = []
    for i in idxs:
        r = df.iloc[i]
        cand.append((f_or(r.get("ae_aesthetic"), -1.0), resolution_score(r), s(r.get("ts")), i))
    cand.sort(key=lambda t: (-t[0], -t[1], t[2]))
    rep_idx = cand[0][3]
    reason = "highest_aesthetic"
    tied = [c for c in cand if c[0] == cand[0][0]]
    if len(tied) > 1:
        tied.sort(key=lambda t: (-t[1], t[2]))
        rep_idx = tied[0][3]
        if tied[0][1] == max(c[1] for c in tied):
            reason = "highest_resolution"
        else:
            reason = "earliest_ts"
    return rep_idx, reason


def atomic_write_csv(df: pd.DataFrame, csv_path: str, backup: bool):
    if backup and os.path.exists(csv_path):
        pd.read_csv(csv_path).to_csv(f"{csv_path}.bak", index=False)
        logging.info("Backed up CSV to %s.bak", csv_path)
    dirn = os.path.dirname(csv_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="images.", suffix=".csv.tmp", dir=dirn)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Auto-mark near/identical duplicates across bands using pHash (metadata-only).")
    ap.add_argument("--metadata", default="metadata/images.csv", help="Path to metadata CSV")
    ap.add_argument("--near-thresh", type=int, default=3, help="Max Hamming distance to consider 'near'")
    ap.add_argument("--prefix-hex", type=int, default=4, help="Leading hex chars for pHash prefix bucketing")
    ap.add_argument("--min-cluster-size", type=int, default=2, help="Only act on clusters with size >= this")
    ap.add_argument("--backup-csv", action="store_true", help="Write a .bak copy before updating")
    ap.add_argument("--dry-run", action="store_true", help="Compute/report but do not modify CSV")
    ap.add_argument("--audit-dir", default="data_stage/dups", help="Directory for audit CSV")
    args = ap.parse_args()

    csv_path = args.metadata
    if not os.path.exists(csv_path):
        logging.error("Metadata CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    if "phash64" not in df.columns or "image_id" not in df.columns or "is_nsfw" not in df.columns:
        logging.error("Required columns missing. Need at least: phash64, image_id, is_nsfw")
        sys.exit(1)

    for col in ["dupe_group", "dupe_flag", "dupe_score_min", "dupe_rep_image",
                "dupe_removed", "dupe_removed_at", "dupe_method", "dupe_params", "dupe_keep_reason"]:
        if col not in df.columns:
            df[col] = ""

    rows: List[Tuple[int, str]] = []
    for i in range(len(df)):
        r = df.iloc[i]
        if to_bool(r.get("is_nsfw")):
            continue
        ph = s(r.get("phash64"))
        if ph:
            rows.append((i, ph))

    if not rows:
        logging.info("No eligible rows for auto-dedup.")
        sys.exit(0)

    buckets: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    k = max(0, args.prefix_hex)
    for i, ph in rows:
        buckets[ph[:k]].append((i, ph))

    parent: Dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for items in tqdm(buckets.values(), desc="Scanning buckets"):
        n = len(items)
        if n < 2:
            continue
        for a in range(n):
            i, ph_i = items[a]
            for b in range(a + 1, n):
                j, ph_j = items[b]
                d = hamming(ph_i, ph_j)
                if d == 0 or (1 <= d <= args.near_thresh):
                    union(i, j)

    comp_members: Dict[int, List[int]] = defaultdict(list)
    for i, _ in rows:
        comp_members[find(i)].append(i)
    clusters = [m for m in comp_members.values() if len(m) >= args.min_cluster_size]
    if not clusters:
        logging.info("No duplicate clusters found (size >= %d).", args.min_cluster_size)
        sys.exit(0)

    updates = []
    now_iso = datetime.utcnow().isoformat() + "Z"

    for idxs in clusters:
        rep_idx, keep_reason = choose_representative(idxs, df)
        rep_row = df.iloc[rep_idx]
        rep_id = s(rep_row.get("image_id"))
        rep_ph = s(rep_row.get("phash64"))
        group_id = f"group_{rep_id}"

        if not args.dry_run:
            df.at[rep_idx, "dupe_group"] = group_id
            df.at[rep_idx, "dupe_keep_reason"] = keep_reason
            df.at[rep_idx, "dupe_method"] = "phash_auto"
            df.at[rep_idx, "dupe_params"] = json.dumps({"near_thresh": args.near_thresh, "prefix_hex": args.prefix_hex})
            df.at[rep_idx, "dupe_removed"] = ""
            df.at[rep_idx, "dupe_removed_at"] = ""
            df.at[rep_idx, "dupe_rep_image"] = rep_id
            df.at[rep_idx, "dupe_flag"] = ""
            df.at[rep_idx, "dupe_score_min"] = "0"

        updates.append({
            "group": group_id, "rep_image_id": rep_id, "image_id": rep_id,
            "dupe_flag": "", "ham_to_rep": 0, "kept": True, "keep_reason": keep_reason
        })

        for i in idxs:
            if i == rep_idx:
                continue
            r = df.iloc[i]
            img_id = s(r.get("image_id"))
            ham = hamming(rep_ph, s(r.get("phash64")))
            dupe_flag = "identical" if ham == 0 else "near"

            if not args.dry_run:
                df.at[i, "dupe_group"] = group_id
                df.at[i, "dupe_flag"] = dupe_flag
                df.at[i, "dupe_score_min"] = str(ham)
                df.at[i, "dupe_rep_image"] = rep_id
                df.at[i, "dupe_removed"] = "TRUE"
                df.at[i, "dupe_removed_at"] = now_iso
                df.at[i, "dupe_method"] = "phash_auto"
                df.at[i, "dupe_params"] = json.dumps({"near_thresh": args.near_thresh, "prefix_hex": args.prefix_hex})

            updates.append({
                "group": group_id, "rep_image_id": rep_id, "image_id": img_id,
                "dupe_flag": dupe_flag, "ham_to_rep": ham, "kept": False, "keep_reason": ""
            })

    os.makedirs(args.audit_dir, exist_ok=True)
    audit_path = os.path.join(args.audit_dir, f"auto_dedup_audit_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")
    pd.DataFrame.from_records(updates, columns=[
        "group", "rep_image_id", "image_id", "dupe_flag", "ham_to_rep", "kept", "keep_reason"
    ]).to_csv(audit_path, index=False)
    logging.info("Audit written to %s", audit_path)

    if args.dry_run:
        logging.info("Dry-run: metadata not modified.")
    else:
        atomic_write_csv(df, csv_path, backup=args.backup_csv)
        logging.info("Metadata updated at %s", csv_path)

    logging.info("Done. Clusters processed: %d", len(clusters))


if __name__ == "__main__":
    main()


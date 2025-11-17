"""
Scores Pokémon fan-art images using GPT-5 (vision) and writes results to metadata/images.csv.

Usage:
  # Batch score from metadata (skips NSFW, resumes where both scores exist)
  python preprocess/score.py --repo-root . --metadata metadata/images.csv --model gpt-5-mini

  # Batch with shuffle + limit + periodic checkpoints
  python preprocess/score.py --repo-root . --metadata metadata/images.csv \
    --model gpt-5-mini --shuffle --limit 500 --checkpoint-every 25 --backup-csv

  # Score a single image without touching the CSV
  python preprocess/score.py --image data_stage/resized/band1/Pikachu/abcd1234.png --dry-run --model gpt-5-mini --verbose

export OPENAI_API_KEY="..."
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import tempfile
import signal
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


FEWSHOT_EXAMPLES = [
    {
        "path": "data_stage/scoring_fs/strong.jpg",
        "user_text": (
            "Example A (training only) — Umbreon with strong grid lines across subject/background. "
            "Artifacts override cuteness/appeal; apply **hard cap** (aesthetic ≤ 15). Output JSON only."
        ),
        "assistant_json": {"r": 95, "a": 10},
    },
    {
        "path": "data_stage/scoring_fs/mild.jpg",
        "user_text": (
            "Example B (training only) — Umbreon with **mild** grid lines (mainly background; still visible across the scene). "
            "Artifacts override appeal; apply **soft cap** (aesthetic ≤ 25). Output JSON only."
        ),
        "assistant_json": {"r": 93, "a": 22},
    },
    {
        "path": "data_stage/scoring_fs/clean.jpg",
        "user_text": (
            "Example C (training only) — clean Umbreon fan-art with **no** grid/tiling/seam artifacts. Output JSON only."
        ),
        "assistant_json": {"r": 95, "a": 93},
    },
]


SYSTEM_TEXT = """You are an image judge scoring Pokémon fan-art for SDXL LoRA (base+refiner) curation.
Follow the rubric strictly. Be deterministic and conservative. Output only JSON—no extra text.

HARD ARTIFACT CAP (takes precedence over all other considerations):
• If any grid/tiling/checkerboard artifact or inpainting seam is visible anywhere in the image:
  – Obvious/strong → set aesthetic ≤ 15
  – Mild/noticeable → set aesthetic ≤ 25
This cap overrides cuteness/appeal and all other criteria.

Relevance (0–100): how clearly the image is Pokémon fan-art (ANY species; multiple species allowed).
• Ignore which species; do not penalize multiple species or background humans.
• Accept stylization, costumes, gijinka/anthro, alt forms, non-canonical styles if Pokémon identity is evident.
• Score from pixels only (ignore filenames, tags, EXIF).

Hard 0 (immediate relevance=0): card images/photos/scans/screenshots (any Pokémon cards/UI/binders/listings); or real-world artifact photos (toys/merch/prints/product photos) where the photographed object, not drawn/painted character art, is the subject.

Relevance guidance (when not Hard 0):
• 1–29: Not evidently Pokémon (generic creature/anime OC/other IP).
• 30–60: Ambiguous Pokémon-inspired cues only.
• 60–85: Clear Pokémon fan-art (may be occluded or heavily stylized).
• 86–100: Unambiguous Pokémon fan-art (any species), group scenes fine.

Aesthetic (0–100) — baseline balance ≈ 60% Technical + 40% Human Appeal.
Technical: composition/framing, lighting/exposure, color harmony, rendering cleanliness, anatomy/proportions, perspective, edge quality, compression/noise, resizing/up/downscale artifacts.
Human Appeal: cuteness/expressiveness, charm, character appeal, mood, storytelling, visual impact.

Bands (after applying the ARTIFACT CAP first):
• 0–15: Obvious grids/tiling/seams (hard cap) OR unusable (severe artifacts).
• 16–29: Mild grids/tiling/seams (soft cap).
• 30–50: Weak/messy; poor composition/lighting or muddy render; limited appeal.
• 51–74: Decent; competent or cute but with issues.
• 75–84: Good; solid technique and/or clear charm.
• 85–92: Strong; clean technique and high appeal.
• 93–100: Exceptional; standout craft and charm; minimal flaws.

Notes:
• Multiple Pokémon is fine; do not downscore crowded scenes if readable.
• Watermarks/signatures: ignore unless they dominate the image.
• If uncertain, be conservative (slightly lower the score).

Return ONLY a single JSON object with integer fields r (relevance) and a (aesthetic). Example: {"r":82,"a":76}
"""

USER_FINAL_REMINDER = """Output JSON ONLY with integer scores:
{"r":<0-100>,"a":<0-100>}
Examples are NOT gold labels; judge from PIXELS ONLY. Apply the ARTIFACT CAP first (≤15 strong, ≤25 mild)."""

JSON_HARD_REQUIREMENT = (
    'Return ONLY a single JSON object with keys r and a (integers 0–100). '
    'No prose. No code fences. No extra keys. Example: {"r":82,"a":76}'
)

def to_data_url(img_path: Path) -> str:
    mime, _ = mimetypes.guess_type(img_path.name)
    if mime is None:
        mime = "image/jpeg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def find_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None

def parse_scores(text: str) -> Optional[Tuple[int, int]]:
    j = find_json_block(text)
    if not j:
        return None
    try:
        obj = json.loads(j.strip("` \n"))
    except Exception:
        return None

    rel = obj.get("relevance", obj.get("r"))
    aes = obj.get("aesthetic", obj.get("a"))
    if rel is None or aes is None:
        return None

    def as_int_0_100(x):
        if isinstance(x, str):
            try:
                x = round(float(x))
            except Exception:
                raise ValueError("Non-numeric")
        if isinstance(x, float):
            x = round(x)
        if not isinstance(x, int):
            raise ValueError("Not int-like")
        return max(0, min(100, x))

    try:
        return as_int_0_100(rel), as_int_0_100(aes)
    except Exception:
        return None

def atomic_write_csv(df: pd.DataFrame, out_path: Path, backup: bool):
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and out_path.exists():
        shutil.copyfile(out_path, str(out_path) + ".bak")
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", suffix=".tmp", dir=str(out_path.parent)) as tf:
        df.to_csv(tf.name, index=False)
        temp_name = tf.name
    shutil.move(temp_name, out_path)

def truthy(val) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y")

def resolve_image_path(row: pd.Series, repo: Path, resized_root: Path) -> Optional[Path]:
    outp = row.get("out_path")
    if isinstance(outp, str) and outp.strip():
        p = Path(outp)
        if not p.is_absolute():
            p = (repo / outp).resolve()
        if p.exists() and p.is_file():
            return p

    image_id = str(row.get("image_id") or "").strip()
    if not image_id:
        return None

    candidates = list(resized_root.glob(f"band*/*/{image_id}.*"))
    if not candidates:
        candidates = list(resized_root.glob(f"**/{image_id}.*"))

    if candidates:
        ranked = sorted(
            candidates,
            key=lambda p: {".png": 0, ".jpg": 1, ".jpeg": 1, ".webp": 2}.get(p.suffix.lower(), 99),
        )
        return ranked[0]
    return None

def build_fewshot_messages(repo_root: Path, enable: bool = True) -> List[dict]:
    if not enable:
        return []
    msgs: List[dict] = []
    for ex in FEWSHOT_EXAMPLES:
        p = Path(ex["path"])
        if not p.is_absolute():
            p = (repo_root / ex["path"]).resolve()
        try:
            data_url = to_data_url(p)
        except Exception:
            # If the image can't be read, skip this few-shot to avoid breaking the run
            continue
        msgs.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": ex["user_text"]},
                {"type": "input_image", "image_url": data_url},
            ],
        })
        msgs.append({
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": json.dumps(ex["assistant_json"], ensure_ascii=False)}
            ],
        })
    return msgs


def _responses_create_with_backoffs(client: OpenAI, payload: dict) -> tuple:
    def _send(pl):
        return client.responses.create(**pl)

    try:
        return _send(payload), False
    except TypeError:
        pl2 = dict(payload)
        pl2.pop("response_format", None)
        try:
            for msg in pl2["input"]:
                if msg.get("role") == "user":
                    for part in msg["content"]:
                        if part.get("type") in ("input_text", "text"):
                            part["text"] = part["text"].rstrip() + "\n\n" + JSON_HARD_REQUIREMENT
                            break
                    break
        except Exception:
            pass
        try:
            return _send(pl2), True
        except BadRequestError as e:
            err = str(e).lower()
            pl3 = dict(pl2)
            changed = False
            if "temperature" in pl3 and ("unsupported parameter" in err or "'temperature'" in err):
                pl3.pop("temperature", None); changed = True
            if "top_p" in pl3 and ("unsupported parameter" in err or "'top_p'" in err):
                pl3.pop("top_p", None); changed = True
            if changed:
                return _send(pl3), True
            raise
    except BadRequestError as e:
        err = str(e).lower()
        pl2 = dict(payload)
        changed = False
        if "temperature" in pl2 and ("unsupported parameter" in err or "'temperature'" in err):
            pl2.pop("temperature", None); changed = True
        if "top_p" in pl2 and ("unsupported parameter" in err or "'top_p'" in err):
            pl2.pop("top_p", None); changed = True
        if changed:
            return client.responses.create(**pl2), True
        raise

def call_model_on_image(
    client: OpenAI,
    model: str,
    data_url: str,
    temperature: float,
    top_p: float,
    repo_root: Path,
    use_fewshot: bool = True,
) -> str:
    
    input_msgs = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_TEXT}]},
    ]
    input_msgs += build_fewshot_messages(repo_root, enable=use_fewshot)
    input_msgs += [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_FINAL_REMINDER},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ]

    payload = dict(
        model=model,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"},
        input=input_msgs,
    )

    resp, _used_fallback = _responses_create_with_backoffs(client, payload)

    text = getattr(resp, "output_text", None)
    if text:
        return text.strip().strip("`")
    try:
        parts = resp.output[0].content
        for p in parts:
            if p.get("type") in ("output_text", "text"):
                return p.get("text", "").strip().strip("`")
    except Exception:
        pass
    return ""

def score_one_image(
    client: OpenAI,
    model: str,
    img_path: Path,
    temperature: float,
    top_p: float,
    repo_root: Path,
    use_fewshot: bool,
    retries: int = 2,
) -> Optional[Tuple[int, int, str]]:
    data_url = to_data_url(img_path)
    last_text = ""
    t = temperature
    for _ in range(retries + 1):
        try:
            text = call_model_on_image(client, model, data_url, t, top_p, repo_root, use_fewshot)
            last_text = text or ""
            parsed = parse_scores(last_text)
            if parsed:
                return parsed[0], parsed[1], last_text
        except Exception as e:
            last_text = f"ERROR: {e}"
        t = max(0.0, min(1.0, (t or 0.0) + 0.05))
    return None


def main():
    ap = argparse.ArgumentParser(description="Score Pokémon fan-art images and update metadata/images.csv")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root (where metadata/ and data_stage/ live)")
    ap.add_argument("--resized-root", type=Path, default=None, help="Root of resized images (default: <repo_root>/data_stage/resized)")
    ap.add_argument("--metadata", type=Path, default=None, help="Path to metadata/images.csv (default: <repo_root>/metadata/images.csv)")
    ap.add_argument("--model", type=str, default="gpt-5", help="Vision-capable model (e.g., gpt-5 or gpt-5-mini)")
    ap.add_argument("--temperature", type=float, default=0.05, help="Sampling temperature (some models ignore/forbid this)")
    ap.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p (some models ignore/forbid this)")
    ap.add_argument("--limit", type=int, default=None, help="Score at most N eligible rows")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle row order before limiting")
    ap.add_argument("--dry-run", action="store_true", help="Do everything but DO NOT write CSV")
    ap.add_argument("--backup-csv", action="store_true", help="Write metadata/images.csv.bak before saving")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--image", type=Path, default=None, help="Score a single image (combine with --dry-run to avoid CSV writes)")
    ap.add_argument("--no-fewshot", action="store_true", help="Disable few-shot examples (default: enabled)")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="Write CSV checkpoint every N successful updates (default 20; 0=only at end)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    resized_root = (args.resized_root or (repo / "data_stage" / "resized")).resolve()
    metadata_csv = (args.metadata or (repo / "metadata" / "images.csv")).resolve()
    use_fewshot = not args.no_fewshot

    if args.image is not None:
        img = args.image.resolve()
        if not img.exists():
            print(f"[ERROR] --image not found: {img}", file=sys.stderr)
            sys.exit(1)

        client = OpenAI()
        res = score_one_image(
            client=client,
            model=args.model,
            img_path=img,
            temperature=args.temperature,
            top_p=args.top_p,
            repo_root=repo,
            use_fewshot=use_fewshot,
        )
        if not res:
            print(f"[PARSE-FAIL] {img}", file=sys.stderr)
            if args.verbose:
                raw = call_model_on_image(client, args.model, to_data_url(img), args.temperature, args.top_p, repo, use_fewshot)
                print("[RAW MODEL OUTPUT]")
                print(raw)
            sys.exit(2)

        rel, aes, raw = res
        if args.verbose:
            print("[RAW MODEL OUTPUT]")
            print(raw)
        print(json.dumps({"r": rel, "a": aes, "image": str(img)}, ensure_ascii=False))

        if not args.dry_run:
            if not metadata_csv.exists():
                print(f"[ERROR] Metadata CSV not found: {metadata_csv}", file=sys.stderr)
                sys.exit(1)
            df = pd.read_csv(metadata_csv)

            if "ae_relevance" not in df.columns:
                df["ae_relevance"] = pd.NA
            if "ae_aesthetic" not in df.columns:
                df["ae_aesthetic"] = pd.NA
            if "path_resized" not in df.columns:
                df["path_resized"] = pd.NA

            stem = img.stem
            row_idx = None
            if "image_id" in df.columns:
                hits = df.index[df["image_id"].astype(str) == stem].tolist()
                if len(hits) == 1:
                    row_idx = hits[0]
            if row_idx is None and "out_path" in df.columns:
                hits = df.index[df["out_path"].astype(str)].str.contains(stem, regex=False, na=False).tolist()  # noqa
            if row_idx is None:
                # fallback search across path-like columns
                candidates = [c for c in df.columns if "path" in c or "file" in c]
                for c in candidates:
                    s = df[c].astype(str)
                    hits = s[s.str.contains(stem, regex=False, na=False)]
                    if len(hits) == 1:
                        row_idx = hits.index[0]
                        break

            if row_idx is None:
                print(f"[MISS] No matching metadata row for {img}", file=sys.stderr)
                sys.exit(0)

            df.at[row_idx, "ae_relevance"] = int(rel)
            df.at[row_idx, "ae_aesthetic"] = int(aes)
            try:
                relpath = str(img.relative_to(repo))
            except Exception:
                relpath = str(img)
            df.at[row_idx, "path_resized"] = relpath

            atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
            print(f"[OK] Updated CSV for {img} -> {metadata_csv}")
        else:
            print("[DRY-RUN] CSV not modified.")
        sys.exit(0)

    if not metadata_csv.exists():
        print(f"[ERROR] Metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv)

    if "ae_relevance" not in df.columns:
        df["ae_relevance"] = pd.NA
    if "ae_aesthetic" not in df.columns:
        df["ae_aesthetic"] = pd.NA
    if "path_resized" not in df.columns:
        df["path_resized"] = pd.NA

    def _is_scored(row) -> bool:
        try:
            ra = row.get("ae_relevance")
            aa = row.get("ae_aesthetic")
            return (pd.notna(ra) and str(ra) != "") and (pd.notna(aa) and str(aa) != "")
        except Exception:
            return False

    def _nsfw_flag(row) -> bool:
        return truthy(row.get("is_nsfw"))

    eligible_idxs = [i for i, row in df.iterrows() if (not _nsfw_flag(row)) and (not _is_scored(row))]

    if not eligible_idxs:
        print("[INFO] No eligible rows (either all NSFW or already scored). Nothing to do.")
        sys.exit(0)

    if args.shuffle:
        random.shuffle(eligible_idxs)
    if args.limit is not None:
        eligible_idxs = eligible_idxs[: args.limit]

    client = OpenAI()

    updated = 0
    skipped_nopath = 0
    parsed_fail = 0

 
    interrupted = {"flag": False}
    def _handle_sig(signum, frame):
        interrupted["flag"] = True
        print("\n[INTERRUPT] Signal received — writing checkpoint...", file=sys.stderr)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_sig)
        except Exception:
            pass  

    def save_checkpoint_if_needed(force: bool = False):
        if args.dry_run:
            return
        if force or (args.checkpoint_every and updated % args.checkpoint_every == 0 and updated > 0):
            atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
            print(f"[CHECKPOINT] wrote CSV after {updated} updates -> {metadata_csv}")

    for row_idx in tqdm(eligible_idxs, desc="Scoring rows"):
        if interrupted["flag"]:
            break

        row = df.iloc[row_idx]
        img_path = resolve_image_path(row, repo, resized_root)
        if img_path is None or not img_path.exists():
            skipped_nopath += 1
            continue

        res = score_one_image(client, args.model, img_path, args.temperature, args.top_p, repo, use_fewshot)
        if not res:
            parsed_fail += 1
            continue

        rel, aes, _ = res
        df.at[row_idx, "ae_relevance"] = int(rel)
        df.at[row_idx, "ae_aesthetic"] = int(aes)
        try:
            relpath = str(img_path.relative_to(repo))
        except Exception:
            relpath = str(img_path)
        df.at[row_idx, "path_resized"] = relpath
        updated += 1

        save_checkpoint_if_needed()

    print(f"[SUMMARY] rows_eligible={len(eligible_idxs)} updated={updated} skipped_nopath={skipped_nopath} parse_fail={parsed_fail}")

    if args.dry_run:
        print("[DRY-RUN] Not writing metadata CSV.")
        return

    atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
    print(f"[OK] Wrote updated CSV: {metadata_csv}")

if __name__ == "__main__":
    main()

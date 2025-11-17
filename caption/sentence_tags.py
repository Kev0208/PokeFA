"""
Generate one-sentence + structured tags for each image in metadata/base.csv.

Usage:
  python caption/sentence_tags.py --repo-root /path/to/PokeFA --metadata metadata/base.csv
  python caption/sentence_tags.py --limit 50 --shuffle
  python catpion/sentence_tags.py --image data_clean/base/abc123.jpg --dry-run

export OPENAI_API_KEY="..."
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from openai import BadRequestError

ART_STYLE_ALLOWED = [
    "anime","chibi","pixel_art","3d_render","watercolor","sketch",
    "lineart","painterly","cel_shade","lineless","realistic","semi_realistic"
]

COMPOSITION_ALLOWED = [
    "close_up","extreme_close_up","portrait","bust","half_body","three_quarter_view","full_body",
    "wide_shot","long_shot","establishing_shot","centered","rule_of_thirds","symmetrical",
    "negative_space","dynamic_angle","dutch_angle","low_angle","high_angle","top_down",
    "isometric","over_the_shoulder","profile_view","back_view","front_view","panoramic",
    "macro","silhouette"
]

SYSTEM_TEXT = f"""You produce HYBRID CAPTIONS for Pokémon fan-art images: ONE sentence + structured tags.

Output JSON ONLY with this shape:
{{
  "sentence": "<12–22 words, present tense>",
  "tags": {{
    "Pokémon": [],              // OPEN: one or more species; use lowercase snake_case names
    "form": [],                 // OPEN: same length as Pokémon; each null or a form like "alolan","mega_x"
    "shiny_presence": false,    // CLOSED boolean; true if ANY shiny appears in the image
    "shiny_pokemon": [],        // OPEN: species that are shiny (subset of Pokémon), e.g., ["charizard"]
    "trainer_presence": false,  // CLOSED boolean; true only if a human trainer is visible
    "gijinka_humanoid": false,  // CLOSED boolean; true only if Pokémon is re-imagined as a human
    "subject_count": "single",  // CLOSED: "single" | "multiple" (pick one)
    "multi_panel": false,       // CLOSED boolean; true only if the image has explicit panels/frames

    "pose": [],                 // OPEN: static body configs, e.g., ["sitting","lying_prone"]
    "actions": [],              // OPEN: dynamic verbs/effects, e.g., ["jumping","flamethrower"]

    "outfit": [],               // OPEN: ["chef_hat","business_suit"], etc.
    "props": [],                // OPEN: ["cake_slice","sparkles"], etc.

    "environment": [],          // OPEN (examples): ["indoor","outdoor","studio","underwater","space","underground"]; other lowercase snake_case values allowed; [] allowed
    "biome": [],                // OPEN (examples): ["mountain","forest","beach","desert","city","cave","snowfield","stadium","ruins"]; other values allowed; [] allowed
    "time_of_day": [],          // OPEN (examples): ["day","dusk","night","sunrise","sunset"]; other values allowed; [] allowed
    "weather": [],              // OPEN (examples): ["clear","rain","snow","fog","storm"]; other values allowed; [] allowed
    "background": [],           // OPEN (examples): ["white","black","patterned","gradient","moon","sun","stars","fireworks","aurora","checkered"]; other values allowed; [] allowed

    "composition": [],          // CLOSED SUBSET of {COMPOSITION_ALLOWED}

    "art_style": [],            // CLOSED SUBSET of {ART_STYLE_ALLOWED}
    "characteristics": []       // OPEN: ["cute","aggressive","happy","angry"], etc.
  }}
}}

Sentence rules:
- 12–25 words (max 28). Present tense. Natural phrasing. No lists or semicolons.
- Order: subject → action → scene/time → composition → (optional ONE style cue).
- Shiny phrasing:
  - If shiny_presence=false → do not mention shiny.
  - If shiny_presence=true and all depicted of one species are shiny → you may start with “Shiny {{species}} …” when that species is the clear focus.
  - If mixed shiny + normal of the same species → write it inline, e.g., “two {{species}}, one shiny” or “one shiny, one normal {{species}}”.
  - If different species and only some are shiny → name the shiny one(s) inline, e.g., “a shiny {{A}} with {{B}}”.
  - Keep it brief; avoid calling shiny if it could be lighting/filters.
- If gijinka_humanoid=true, start with “Humanized {{species}} …”.
- If trainer_presence=true and a trainer is clearly depicted: “A trainer with {{primary species}} …” (no invented names).
- If subject_count="multiple" and all subjects are same species: “Multiple {{species}} …”; for different species, “{{A}} and {{B}} …”.
- Mention at most ONE central prop or outfit if visually important.
- For studio/white backgrounds, you may say “on white” or “clean background”.
- DO NOT enumerate tags; write a fluent single sentence.

Tag rules:
- Use lowercase snake_case for all open-string tokens.
- Pokémon & form must be the SAME LENGTH (1:1); use null when no form applies.
- art_style MUST be a subset of this CLOSED list: {ART_STYLE_ALLOWED}.
- Be conservative with booleans; only set true when visually unambiguous.
- Leave arrays empty if unsure (do NOT invent).
- For OPEN facets (environment, biome, time_of_day, weather, background), the example lists are NOT exhaustive; you may use other concise, descriptive tokens. Empty [] is allowed when information is not clearly visible.
- For the CLOSED composition facet, use only the allowed values and pick the most salient 0–2 terms.
"""

USER_MSG = "Return ONLY one JSON object with 'sentence' and 'tags' per the schema. No extra text."

def to_data_url(img_path: Path) -> str:
    mime, _ = mimetypes.guess_type(img_path.name)
    if mime is None:
        mime = "image/jpeg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def atomic_write_csv(df: pd.DataFrame, out_path: Path, backup: bool):
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and out_path.exists():
        shutil.copyfile(out_path, str(out_path) + ".bak")
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", suffix=".tmp", dir=str(out_path.parent)) as tf:
        df.to_csv(tf.name, index=False)
        tmp = tf.name
    shutil.move(tmp, out_path)

_json_block_re = re.compile(r"\{[\s\S]*\}\s*$")

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _json_block_re.search(text.strip().strip("`"))
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def sanitize_sentence(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().strip("`").strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    parts = re.split(r'(?<=[.!?])\s+', s)
    if parts:
        s = parts[0].strip()
    if not re.search(r"[.!?]$", s):
        s += "."
    words = s.split()
    if len(words) > 28:
        s = " ".join(words[:28]).rstrip(",;") + "."
    return s

def coerce_schema(obj: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], str]]:
    err = []
    if not isinstance(obj, dict):
        return None

    # Sentence
    sentence = sanitize_sentence(obj.get("sentence", ""))

    tags = obj.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}

    legacy_bool_map = {
        "shiny": "shiny_presence",
        "shiny_present": "shiny_presence",
        "trainer_present": "trainer_presence",
    }
    for old, new in legacy_bool_map.items():
        if old in tags and new not in tags:
            try:
                tags[new] = bool(tags[old])
            except Exception:
                tags[new] = False

    for k in ("shiny", "shiny_present", "trainer_present"):
        if k in tags:
            tags.pop(k, None)

    required = [
        "Pokémon", "form",
        "shiny_presence", "shiny_pokemon", "trainer_presence", "gijinka_humanoid",
        "subject_count", "multi_panel",
        "pose", "actions", "outfit", "props",
        "environment", "biome", "time_of_day", "weather", "background",
        "composition", "art_style", "characteristics",
    ]
    list_keys = {"Pokémon","form","shiny_pokemon","pose","actions","outfit","props",
                 "environment","biome","time_of_day","weather","background",
                 "composition","art_style","characteristics"}
    for k in required:
        if k in list_keys:
            tags.setdefault(k, [])
        else:
            tags.setdefault(k, False)

    for k in ("Pokémon", "form"):
        v = tags.get(k)
        if isinstance(v, str) or v is None:
            tags[k] = [v]
        elif not isinstance(v, list):
            tags[k] = [str(v)]

    p_len = len(tags["Pokémon"])
    f_len = len(tags["form"])
    if f_len < p_len:
        tags["form"].extend([None] * (p_len - f_len))
    elif f_len > p_len:
        tags["form"] = tags["form"][:p_len]
        err.append("trim_form_len")

    if isinstance(tags.get("art_style"), list):
        tags["art_style"] = [s for s in tags["art_style"] if isinstance(s, str) and s in ART_STYLE_ALLOWED]
    else:
        tags["art_style"] = []

    if isinstance(tags.get("composition"), list):
        tags["composition"] = [s for s in tags["composition"] if isinstance(s, str) and s in COMPOSITION_ALLOWED][:2]
    else:
        tags["composition"] = []

    if tags.get("subject_count") not in ("single", "multiple"):
        tags["subject_count"] = "single"
    tags["multi_panel"] = bool(tags.get("multi_panel", False))
    for b in ("shiny_presence", "trainer_presence", "gijinka_humanoid"):
        tags[b] = bool(tags.get(b, False))

    def snakeify_list(vals):
        out = []
        for v in vals or []:
            if v is None:
                out.append(None)
                continue
            s = str(v).strip().lower()
            s = re.sub(r"[^a-z0-9]+", "_", s)
            s = re.sub(r"_+", "_", s).strip("_")
            if s:
                out.append(s)
        seen = set()
        uniq = []
        for x in out:
            if x is None or x not in seen:
                uniq.append(x)
                if x is not None:
                    seen.add(x)
        return uniq

    for k in ("Pokémon","shiny_pokemon","pose","actions","outfit","props",
              "environment","biome","time_of_day","weather","background","characteristics"):
        if isinstance(tags.get(k), list):
            tags[k] = snakeify_list(tags[k])
        else:
            tags[k] = []

    forms = tags.get("form", [])
    fixed_forms = []
    for f in forms:
        if f is None:
            fixed_forms.append(None)
        else:
            s = str(f).strip().lower()
            s = re.sub(r"[^a-z0-9]+", "_", s)
            s = re.sub(r"_+", "_", s).strip("_")
            fixed_forms.append(s if s else None)
    tags["form"] = fixed_forms

    return sentence, tags, "|".join(err)



def responses_create(client: OpenAI, payload: dict):
    """Create a response with light fallbacks for param compatibility."""
    def _send(pl):
        return client.responses.create(**pl)
    try:
        return _send(payload), False
    except TypeError:
        pl2 = dict(payload)
        pl2.pop("response_format", None)
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

def call_model(client: OpenAI, model: str, data_url: str, temperature: float, top_p: float):
    msgs = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": [
            {"type": "input_text", "text": USER_MSG},
            {"type": "input_image", "image_url": data_url},
        ]},
    ]
    payload = dict(model=model, temperature=temperature, top_p=top_p,
                   response_format={"type":"json_object"}, input=msgs)

    try:
        resp, _ = responses_create(client, payload)
    except Exception as e:
        return None, f"[API_ERR] {e!r}"

    txt = getattr(resp, "output_text", None)
    if txt:
        return extract_json(txt), txt

    try:
        parts = resp.output[0].content
        for p in parts:
            if p.get("type") in ("output_text","text"):
                raw = p.get("text","")
                return extract_json(raw), raw
    except Exception as e:
        return None, f"[EXTRACT_ERR] {e!r} | raw_obj={getattr(resp, 'to_dict', lambda: str(resp))()}"

    return None, "[EMPTY_OUTPUT]"


def caption_tags_for_image(client, model, img_path, temperature, top_p, retries: int = 2):
    data_url = to_data_url(img_path)
    last_raw = ""
    t = temperature
    for _ in range(retries + 1):
        obj, raw = call_model(client, model, data_url, t, top_p)
        last_raw = raw or last_raw
        if obj:
            sentence, tags, err = coerce_schema(obj)
            if sentence and isinstance(tags, dict):
                return sentence, tags, err
        t = max(0.0, min(1.0, (t or 0.0) + 0.05))
    # surface why it failed
    return None, last_raw

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Generate hybrid captions (sentence + tags) and update metadata/base.csv")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root (where metadata/ and data_clean/ live)")
    ap.add_argument("--metadata", type=Path, default=None, help="Path to metadata/base.csv (default: <repo_root>/metadata/base.csv)")
    ap.add_argument("--model", type=str, default="gpt-5-mini", help="Vision-capable model (default: gpt-5-mini)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N rows")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle row order before limiting")
    ap.add_argument("--dry-run", action="store_true", help="Do everything but DO NOT write CSV")
    ap.add_argument("--backup-csv", action="store_true", help="Write metadata/base.csv.bak before saving")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--image", type=Path, default=None, help="Caption a single image (prints JSON). Combine with --dry-run to avoid CSV writes")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="Write CSV checkpoint every N successful updates (0=only at end)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty caption/tags")
    ap.add_argument("--caption-col", type=str, default="caption", help="CSV column name for the sentence")
    ap.add_argument("--tags-col", type=str, default="tags", help="CSV column name for the tags JSON")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    metadata_csv = (args.metadata or (repo / "metadata" / "base.csv")).resolve()

    if args.image is not None:
        img = args.image.resolve()
        if not img.exists():
            print(f"[ERROR] --image not found: {img}", file=sys.stderr)
            sys.exit(1)

        client = OpenAI()
        res = caption_tags_for_image(client, args.model, img, args.temperature, args.top_p)
        if not res or res[0] is None:
            raw = res[1] if res else None
            print(f"[PARSE-FAIL] {img}", file=sys.stderr)
            if args.verbose and raw:
                print(f"[RAW] {raw}", file=sys.stderr)
            sys.exit(2)
        sentence, tags, _ = res
        out = {"sentence": sentence, "tags": tags}
        print(json.dumps(out, ensure_ascii=False))
        if args.dry_run:
            print("[DRY-RUN] CSV not modified.")
            sys.exit(0)

        if not metadata_csv.exists():
            print(f"[WARN] Metadata CSV not found: {metadata_csv} — exiting after single-image caption.", file=sys.stderr)
            sys.exit(0)

        df = pd.read_csv(metadata_csv)
        if "path" not in df.columns:
            print("[ERROR] metadata/base.csv must contain a 'path' column.", file=sys.stderr)
            sys.exit(1)
        if args.caption_col not in df.columns:
            df[args.caption_col] = pd.NA
        if args.tags_col not in df.columns:
            df[args.tags_col] = pd.NA

        try:
            rel = str(img.relative_to(repo))
        except Exception:
            rel = None

        row_idx = None
        s = df["path"].astype(str)
        if rel:
            hits = s[s == rel]
            if len(hits) == 1:
                row_idx = hits.index[0]
        if row_idx is None:
            stem = img.stem
            hits = s[s.str.contains(stem, regex=False, na=False)]
            if len(hits) == 1:
                row_idx = hits.index[0]

        if row_idx is None:
            print(f"[MISS] No matching row for {img}", file=sys.stderr)
            sys.exit(0)

        if (not args.overwrite) and pd.notna(df.at[row_idx, args.caption_col]) and str(df.at[row_idx, args.caption_col]).strip():
            print(f"[SKIP] Row already has {args.caption_col} (use --overwrite): {df.at[row_idx, args.caption_col]}")
            sys.exit(0)

        df.at[row_idx, args.caption_col] = sentence
        df.at[row_idx, args.tags_col] = json.dumps(tags, ensure_ascii=False)
        atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
        print(f"[OK] Updated CSV for {img} -> {metadata_csv}")
        sys.exit(0)

    # ---------- Batch mode ----------
    if not metadata_csv.exists():
        print(f"[ERROR] Metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv)
    if "path" not in df.columns:
        print("[ERROR] metadata/base.csv must contain a 'path' column.", file=sys.stderr)
        sys.exit(1)

    if args.caption_col not in df.columns:
        df[args.caption_col] = pd.NA
    if args.tags_col not in df.columns:
        df[args.tags_col] = pd.NA

    def _needs(idx) -> bool:
        if args.overwrite:
            return True
        c = df.at[idx, args.caption_col]
        t = df.at[idx, args.tags_col]
        return not ((pd.notna(c) and str(c).strip()) and (pd.notna(t) and str(t).strip()))

    eligible_idxs: List[int] = [i for i in df.index.tolist() if _needs(i)]
    if not eligible_idxs:
        print("[INFO] No eligible rows (captions+tags present and --overwrite not set).")
        sys.exit(0)

    if args.shuffle:
        import random
        random.shuffle(eligible_idxs)
    if args.limit is not None:
        eligible_idxs = eligible_idxs[: args.limit]

    client = OpenAI()

    updated = 0
    missing = 0
    parse_fail = 0

    interrupted = {"flag": False}
    def _handle_sig(signum, frame):
        interrupted["flag"] = True
        print("\n[INTERRUPT] Signal received — writing checkpoint...", file=sys.stderr)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_sig)
        except Exception:
            pass

    def checkpoint(force=False):
        if args.dry_run:
            return
        if force or (args.checkpoint_every and updated > 0 and updated % args.checkpoint_every == 0):
            atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
            print(f"[CHECKPOINT] wrote CSV after {updated} updates -> {metadata_csv}")

    for idx in tqdm(eligible_idxs, desc="Captioning rows"):
        if interrupted["flag"]:
            break
        rel_path = str(df.at[idx, "path"]).strip()
        if not rel_path:
            missing += 1
            continue
        img_path = (repo / rel_path).resolve()
        if not img_path.exists():
            missing += 1
            if args.verbose:
                print(f"[MISS] Image not found: {img_path}", file=sys.stderr)
            continue

        res = caption_tags_for_image(client, args.model, img_path, args.temperature, args.top_p)
        if not res:
            parse_fail += 1
            if args.verbose:
                print(f"[PARSE-FAIL] {img_path}", file=sys.stderr)
            continue

        sentence, tags, _ = res
        df.at[idx, args.caption_col] = sentence
        df.at[idx, args.tags_col] = json.dumps(tags, ensure_ascii=False)
        updated += 1
        checkpoint(force=False)

    print(f"[SUMMARY] rows_eligible={len(eligible_idxs)} updated={updated} missing_images={missing} parse_fail={parse_fail}")

    if args.dry_run:
        print("[DRY-RUN] Not writing metadata CSV.")
        return

    atomic_write_csv(df, metadata_csv, backup=args.backup_csv)
    print(f"[OK] Wrote updated CSV: {metadata_csv}")

if __name__ == "__main__":
    main()

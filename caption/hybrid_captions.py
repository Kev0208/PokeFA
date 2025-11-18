"""
Build CLIP-ready hybrid captions: keep sentence, append values-only tokens in priority order.

Usage:
  python caption/hybrid_captions.py --metadata metadata/base.csv --out metadata/base_with_hybrid.csv --col-name hybrid_caption
  python caption/hybrid_captions.py --metadata metadata/base.csv --backup  # overwrite input by default
"""

import argparse, json, os, re, sys, shutil, tempfile
import pandas as pd
from typing import Any, Dict, List, Optional

FORM_PREFIXES = {"alolan":"alolan","galarian":"galarian","hisui":"hisui","hisuian":"hisui","paldean":"paldean",
                 "totem":"totem","origin":"origin","gigantamax":"gigantamax","primal":"primal","shadow":"shadow",
                 "apex":"apex","female":"female","male":"male","therian":"therian","incarnate":"incarnate",
                 "sky":"sky","black":"black","white":"white"}
MEGA_FORMS={"mega","mega_x","mega_y"}; MEGA_SUFFIX={"mega_x":" x","mega_y":" y"}

def _safe_json(s: Any)->Optional[Dict[str,Any]]:
    if s is None or (isinstance(s,float) and pd.isna(s)) or not isinstance(s,str): return None
    try: return json.loads(s)
    except Exception:
        try: return json.loads(s.replace('""','"').strip().rstrip(",")) 
        except Exception: return None

def _norm(s: Any)->str:
    if s is None: return ""
    s = re.sub(r"_+","_",re.sub(r"[^a-z0-9]+","_",str(s).strip().lower())).strip("_")
    return s

def _merge_form(species: str, form: Optional[str], shiny: bool)->str:
    sp = _norm(species).replace("_"," ")
    f = _norm(form) if form else ""
    pref = "shiny " if shiny else ""
    if f in MEGA_FORMS: return f"{pref}mega {sp}{MEGA_SUFFIX.get(f,'')}".strip()
    if f in FORM_PREFIXES: return f"{pref}{FORM_PREFIXES[f]} {sp}".strip()
    return f"{pref}{(f+' '+sp).strip()}" if f else f"{pref}{sp}"

def species_tokens(tags: Dict[str,Any])->List[str]:
    pokemon = tags.get("Pokémon") or tags.get("pokemon") or []
    forms = tags.get("form") or []
    shiny_presence = bool(tags.get("shiny_presence", False))
    shiny_list = tags.get("shiny_pokemon") or []
    if isinstance(pokemon,str): pokemon=[pokemon]
    if isinstance(forms,str) or forms is None: forms=[forms]
    if len(forms)<len(pokemon): forms += [None]*(len(pokemon)-len(forms))
    elif len(forms)>len(pokemon): forms = forms[:len(pokemon)]
    shiny_set = { _norm(x) for x in (shiny_list or []) if x }
    out, seen = [], set()
    for sp, fm in zip(pokemon, forms):
        spn = _norm(sp)
        tok = _merge_form(spn, fm, shiny_presence and ((not shiny_set) or spn in shiny_set))
        if tok and tok not in seen: out.append(tok); seen.add(tok)
    return out

def list_vals(tags: Dict[str,Any], key: str)->List[str]:
    vals = tags.get(key) or []
    if isinstance(vals,str): vals=[vals]
    out, seen = [], set()
    for v in vals:
        t=_norm(v)
        if t and t not in seen: out.append(t); seen.add(t)
    return out

def compose(sentence: str, tags: Optional[Dict[str,Any]])->str:
    sent = (sentence or "").strip()
    if not tags or not isinstance(tags,dict): return sent
    pieces=[]
    pieces += species_tokens(tags)
    pieces += list_vals(tags,"composition")
    pieces += list_vals(tags,"art_style")
    pieces += list_vals(tags,"actions")
    pieces += list_vals(tags,"pose")
    pieces += list_vals(tags,"environment")
    pieces += list_vals(tags,"biome")
    pieces += list_vals(tags,"time_of_day")
    pieces += list_vals(tags,"weather")
    pieces += list_vals(tags,"background")
    pieces += list_vals(tags,"outfit")
    pieces += list_vals(tags,"props")
    pieces += list_vals(tags,"characteristics")
    tail=[]
    sc=_norm(tags.get("subject_count"))
    if sc in ("single","multiple"): tail.append(sc)
    if bool(tags.get("trainer_presence",False)): tail.append("trainer")
    if bool(tags.get("gijinka_humanoid",False)): tail.append("gijinka")
    if bool(tags.get("multi_panel",False)): tail.append("multi_panel")
    pieces += tail
    tokens=", ".join([t for t in pieces if t])
    return f"{sent} | {tokens}" if tokens else sent

def _atomic_write_csv(df, path, backup: bool):
    if backup and os.path.exists(path): shutil.copyfile(path, path+".bak"); print(f"[BACKUP] {path}.bak")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, newline="", suffix=".tmp",
                                         dir=os.path.dirname(path) or ".") as tf:
            df.to_csv(tf.name, index=False); tmp=tf.name
        shutil.move(tmp, path)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

def main():
    ap = argparse.ArgumentParser(description="Make CLIP-ready hybrid captions.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--caption-col", default="caption")
    ap.add_argument("--tags-col", default="tags")
    ap.add_argument("--col-name", default="hybrid_caption")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    in_path = args.metadata
    out_path = args.out or in_path
    if not os.path.exists(in_path): print(f"[ERROR] {in_path} not found", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    for c in (args.caption_col, args.tags_col):
        if c not in df.columns: print(f"[ERROR] CSV missing '{c}'", file=sys.stderr); sys.exit(2)
    if args.col_name not in df.columns: df[args.col_name] = ""
    df[args.col_name] = df[args.col_name].astype("string").fillna("")

    hybrids=[]
    for i in range(len(df)):
        sent = df.at[i, args.caption_col]
        tags = _safe_json(df.at[i, args.tags_col])
        hybrids.append(compose(sent, tags))
    df[args.col_name] = hybrids

    _atomic_write_csv(df, out_path, backup=args.backup)
    print(f"[OK] wrote: {out_path} (column: {args.col_name})")

if __name__ == "__main__":
    main()

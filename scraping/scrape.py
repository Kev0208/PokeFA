"""
Scrape Pokémon fanart images per band/species via SerpAPI Google Images.

Usage:
  python scraping/scrape.py --root . --quotas quotas/quotas.csv --band band1 --api-key <KEY>
  python scraping/scrape.py --root . --quotas quotas/quotas.csv --species "Umbreon" --api-key <KEY>
  python scraping/scrape.py --root . --quotas quotas/quotas.csv --band all --rps 1.5 --max-pages 50 --safe active --api-key <KEY>
"""

from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm

from serpapi_client import SerpApiImages
from utils import (
    append_jsonl,
    append_image_row,
    choose_ext_from_url_or_fmt,
    domain_from_url,
    download_image,
    ensure_root_skeleton,
    ensure_species_dirs,
    images_csv_path,
    iso_now,
    load_quotas,
    make_scrape_run_id,
    md5_bytes,
    safe_species_folder_name,
    urls_jsonl_path,
    UrlSeen,
    validate_image_bytes,
)

def build_query(species_name: str) -> str:
    name = " ".join(str(species_name).split())
    return f"{name} Fanart"

def scrape_species(
    root: Path,
    api: SerpApiImages,
    species_name: str,
    ranking: int,
    band_label: str,
    quota: int,
    scrape_run_id: str,
    max_pages: int = 50,
    safe: str = "active",
    resume: bool = True,
    min_side: int = 128,
) -> int:
    species_folder = safe_species_folder_name(ranking, species_name)
    _, imgs_dir = ensure_species_dirs(root, band_label, species_folder)
    urls_path = urls_jsonl_path(root, band_label, species_folder)
    imgs_csv = images_csv_path(root)

    existing = sum(1 for _ in imgs_dir.glob("*")) if resume else 0
    need = max(0, quota - existing)
    if need <= 0:
        return 0

    seen = UrlSeen()
    saved = 0
    query = build_query(species_name)

    pbar = tqdm(total=need, desc=f"{species_name} ({band_label})", leave=False)
    try:
        for item in api.iter_image_results(query=query, max_pages=max_pages, safe=safe):
            url = item.get("original") or item.get("thumbnail") or item.get("link")
            if not url or not seen.add(url):
                continue

            append_jsonl(urls_path, {
                "status": "attempt",
                "url": url,
                "species": species_name,
                "rank": ranking,
                "band": band_label,
                "query": query,
                "scrape_run_id": scrape_run_id
            })

            dl = download_image(url)
            if not dl:
                append_jsonl(urls_path, {"status": "http_fail", "url": url, "scrape_run_id": scrape_run_id})
                continue
            blob, ctype = dl

            v = validate_image_bytes(blob, min_side=min_side)
            if not v:
                append_jsonl(urls_path, {"status": "invalid_or_corrupt", "url": url, "scrape_run_id": scrape_run_id})
                continue
            fmt, w, h = v

            hsh = md5_bytes(blob)
            ext = choose_ext_from_url_or_fmt(url, fmt)
            out_path = imgs_dir / f"{hsh}{ext}"
            if out_path.exists():
                append_jsonl(urls_path, {
                    "status": "duplicate_existing",
                    "url": url,
                    "hash": hsh,
                    "path": str(out_path),
                    "fmt": fmt, "w": w, "h": h,
                    "scrape_run_id": scrape_run_id
                })
                continue

            try:
                out_path.write_bytes(blob)
                saved += 1
                pbar.update(1)
            except Exception as e:
                append_jsonl(urls_path, {"status": "io_error", "url": url, "error": str(e), "scrape_run_id": scrape_run_id})
                continue

            append_jsonl(urls_path, {
                "status": "saved",
                "url": url,
                "hash": hsh,
                "path": str(out_path),
                "fmt": fmt, "w": w, "h": h,
                "content_type": ctype,
                "scrape_run_id": scrape_run_id
            })

            append_image_row(imgs_csv, {
                "ts": iso_now(),
                "scrape_run_id": scrape_run_id,
                "image_id": hsh,
                "species": species_name,
                "rank": ranking,
                "band": band_label,
                "query": query,
                "source_url": url,
                "source_domain": domain_from_url(url),
                "path_raw": str(out_path),
                "fmt": fmt,
                "w": w,
                "h": h,
                "content_type": ctype,
                "safe": safe,
                "hl": api.hl,
                "gl": api.gl,
            })

            if saved >= need:
                break
    finally:
        pbar.close()

    return saved

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape images into data_raw/ by band/species per quotas and write metadata.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--quotas", required=True)
    parser.add_argument("--band", default="all", choices=["all","band1","band2","band3","band4"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--limit-per-species", type=int, default=None)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--rps", type=float, default=1.5)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--safe", default="active", choices=["active","off"])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--min-side", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.root)
    ensure_root_skeleton(root)

    quotas_path = Path(args.quotas)
    if not quotas_path.exists():
        raise FileNotFoundError(f"Missing quotas CSV: {quotas_path}")

    df = load_quotas(quotas_path)
    if args.band != "all":
        df = df[df["band_label"] == args.band].copy()
    if args.species:
        df = df[df["Name"] == args.species].copy()
    if args.limit_per_species is not None:
        df["q_r_base"] = df["q_r_base"].clip(upper=args.limit_per_species)
    if df.empty:
        print("No species selected.")
        return

    scrape_run_id = make_scrape_run_id()
    api = SerpApiImages(api_key=args.api_key, rps=args.rps)

    total_target = int(df["q_r_base"].sum())
    total_saved = 0
    for _, row in df.iterrows():
        total_saved += scrape_species(
            root=root,
            api=api,
            species_name=row["Name"],
            ranking=int(row["Ranking"]),
            band_label=row["band_label"],
            quota=int(row["q_r_base"]),
            scrape_run_id=scrape_run_id,
            max_pages=args.max_pages,
            safe=args.safe,
            resume=(not args.no_resume),
            min_side=args.min_side,
        )

    print(f"[DONE] species={len(df)} | saved={total_saved}/{total_target}")

if __name__ == "__main__":
    main()

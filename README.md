# PokeFA — Large-scale Pokémon fan-art dataset with relevance/aesthetic scores and hybrid captions

PokeFA is the first large-scale Pokémon fan-art image dataset and an end-to-end, auditable pipeline.  
I collected 30,000 images across 1,025 Pokémon using a popularity-banded budget, then ran an extensive preprocessing pipeline:

NSFW filtering → OCR localization & inpainting → resizing → relevance & aesthetic scoring (OpenAI `gpt-5-mini`) → near-duplicate removal → quality filtering to the top 16,000 images → hybrid captions (sentence + tags via `gpt-5-mini`) → caption cleaning → WebDataset packaging.

Every sample is traceable: I preserve source URLs and a per-image ledger of transformations and decisions.  
➡️ **Full dataset on Hugging Face**: `{repo_link_to_be_provided}`

---

## Repository Layout

```
PokeFA/
├── LICENSE
├── README.md
├── caption/
│   ├── cap_filter.py
│   ├── hybrid_captions.py        
│   ├── recap_merge.py
│   └── sentence_tags.py
├── preprocess/
│   ├── dup/
│   │   ├── phash.py
│   │   └── rm_dups.py
│   ├── filter/
│   │   └── filter.py
│   ├── nsfw/
│   │   ├── nsfw_filter.py
│   │   └── nsfw_unflag.py
│   ├── ocr/
│   │   ├── ocr_pipeline.py
│   │   └── ocr_unflag.py
│   ├── resize/
│   │   └── resize.py
│   └── score/
│       └── score.py
├── quotas/
│   └── quotas.csv
├── requirements/
│   ├── requirements_caption.txt
│   ├── requirements_dup.txt
│   ├── requirements_filter.txt
│   ├── requirements_nsfw.txt
│   ├── requirements_ocr.txt
│   ├── requirements_resize.txt
│   ├── requirements_score.txt
│   └── requirements_scraping.txt
├── scraping/
│   ├── scrape.py
│   ├── serpapi_client.py
│   └── utils.py
└── webdataset/
    ├── species.py
    ├── split.py
    └── webdataset.py
```

> Note on dependencies: Each pipeline stage has its own requirements file to avoid conflicts (e.g., DeepDanbooru v3 for NSFW uses TensorFlow, while OCR/inpainting/scoring rely on PyTorch and other libs). Install per-stage as needed.

---

## Pipeline

**Important:** *All example usages live in each script’s* **top-of-file docstring**. Run scripts directly; follow the documented CLI flags there.

### 0) Popularity-Banded Quotas 
- What: Allocate collection budget across 1,025 Pokémon using rank bands + floors to emphasize popular species while preserving tail coverage.
- Inputs: `quotas/quotas.csv` (derived from the **Thomas Game Docs** popularity survey; ~1.6M responses: https://thomasgamedocs.com/pokemon/).

---

### 1) Raw Data Collection (Scraping)
- Goal: Gather ~30k candidate images by species according to quotas; record source URLs and bytes-level integrity.
- Order:
  1. `scraping/scrape.py` (uses `serpapi_client.py`, `utils.py` under the hood)
- Outputs: `metadata/images.csv` (global ledger), raw images under `data_raw/...`.

---

### 2) NSFW Screening
- Goal: Flag/remove sexual content before any downstream compute or scoring.
- Order:
  1. `preprocess/nsfw/nsfw_filter.py` (automatic tagging/flagging)
  2. `preprocess/nsfw/nsfw_unflag.py` (optional manual review/unflag by ID)
- Outputs: Updated flags/fields in `metadata/images.csv`, audit rows for drops.

---

### 3) OCR Detection & Inpainting
- Goal: Localize text/watermarks/signatures with PaddleOCR, generate masks, and inpaint with LaMa to remove text while preserving composition.
- Order:
  1. `preprocess/ocr/ocr_pipeline.py` (PaddleOCR detect → mask JSON/PNG → LaMa inpaint)
  2. `preprocess/ocr/ocr_unflag.py` (optional manual overrides)
- Outputs: Masks/boxes under `data_stage/OCR_*`, LaMa-inpainted PNGs under `data_stage/OCR_inpainted`, paths/flags recorded in metadata.

---

### 4) Resizing
- Goal: Normalize scale while preserving line quality (bicubic for modest changes, area/box for large shrinks).
- Order:
  1. `preprocess/resize/resize.py`
- Outputs: `data_stage/resized/...`, size/kernel/provenance columns updated in metadata.

---

### 5) Relevance & Aesthetic Scoring
- Goal: Assign Relevance (0–100) and Aesthetic (0–100) using OpenAI `gpt-5-mini` with a strict JSON rubric and artifact caps.
- Order:
  1. `preprocess/score/score.py`
- Outputs: `ae_relevance`, `ae_aesthetic` in `metadata/images.csv` (resume-safe).

---

### 6) Near-Duplicate Removal
- Goal: Collapse near-identical variants using pHash clustering and keep the highest quality (aesthetic → resolution → timestamp).
- Order:
  1. `preprocess/dup/phash.py` (compute perceptual hashes)
  2. `preprocess/dup/rm_dups.py` (cluster & mark duplicates)
- Outputs: `dupe_*` annotations in metadata; audit CSVs; no destructive deletes.

---

### 7) Quality Filtering → Final Base Dataset (16k)
- Goal: From post-NSFW/ocr/resize/scored/dedup data, keep the top 16,000 by gates and rank.
- Order:
  1. `preprocess/filter/filter.py` (hard gates on relevance/aesthetic + ranking; exports `metadata/base.csv` and `data_clean/base/...`)
- Outputs: `metadata/base.csv` with compact schema (`path, source_url, image_size, ae_relevance, ae_aesthetic, image_id, phash`).

---

### 8) Captioning (Hybrid: sentence + tags)
- Goal: Produce clean, informative captions blending natural sentences and structured tags.
- Order:
  1. `caption/sentence_tags.py` (generate sentence + tags with `gpt-5-mini`)
  2. `caption/cap_filter.py` (filter anomalies like unidentified Pokémon for re-captioning)
  3. `caption/recap_merge.py` (merge re-captions back)
  4. `caption/hybrid_captions.py` (compose final hybrid captions)
- Outputs: `hybrid_caption` column attached to base metadata.

---

### 9) WebDataset Conversion
- Goal: Species normalization, split, and tar-shard packaging for efficient ML training.
- Order:
  1. `webdataset/species.py` (normalize species names; drop anomaly tokens; write per-image `species` to `base.csv`; emit global counts)
  2. `webdataset/split.py` (head-species-aware train/val split; writes split tags/manifests)
  3. `webdataset/webdataset.py` (pack `{.jpg/.png, .txt, .json}` per image into train/val shards; emit manifests, id lists, checksums)
- Outputs:
  - `webdataset/data/train-*.tar`, `webdataset/data/val-*.tar`
  - `webdataset/manifests/{split_stats.json, train_species.csv, val_species.csv, train_ids.txt, val_ids.txt}`

---

## Acknowledgements

- Community artists (fan-art creators).
- Thomas Game Docs (popularity survey informing quotas).
- OpenAI (`gpt-5-mini`), PaddleOCR, LaMa inpainting, DeepDanbooru, WebDataset tooling, and the open-source ecosystem.




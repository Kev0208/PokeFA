"""
OCR removal for SDXL fine-tuning: PaddleOCR detect → mask → LaMa inpaint.

Usage:
  # 1) Detect text and write masks + boxes; update CSV columns
  python preprocess/ocr_pipeline.py detect \
    --metadata metadata/images.csv \
    --repo-root . \
    --output-root data_stage/OCR_detection \
    --min-conf 0.35 --min-area-px 150 --mask-dilate 3 \
    --rotate "0,90" --upscale-short-side 1024 \
    --rec-check-thresh 0.0

  # 2) Inpaint masked regions with LaMa (vendored under third_party/lama)
  python preprocess/ocr_pipeline.py inpaint \
    --input-root  data_stage/OCR_detection \
    --output-root data_stage/OCR_inpainted \
    --lama-config     weights/lama/big-lama/config.yaml \
    --lama-checkpoint weights/lama/big-lama/models/best.ckpt
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

# Vendored LaMa (must exist)
import sys as _sys
VENDOR_LAMA_DIR = Path(__file__).resolve().parents[1] / "third_party" / "lama"
_sys.path.insert(0, str(VENDOR_LAMA_DIR))
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_image_with_webp_flatten(img_path: Path) -> Tuple[np.ndarray, bool]:
    ext = img_path.suffix.lower()
    if ext == ".webp":
        try:
            im = Image.open(str(img_path))
            if getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1:
                im.seek(0)
                rgb = im.convert("RGB")
                arr = np.array(rgb)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), True
        except Exception:
            pass
    return cv2.imread(str(img_path)), False

def init_paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="en", use_angle_cls=True)

def ocr_call(ocr, image_bgr: np.ndarray, det=True, rec=False):
    return ocr.ocr(image_bgr, det=det, rec=rec, cls=False)

def parse_conf(score_field) -> float:
    if isinstance(score_field, (float, int)): return float(score_field)
    if isinstance(score_field, (list, tuple)) and len(score_field) >= 2 and isinstance(score_field[1], (float,int)):
        return float(score_field[1])
    if isinstance(score_field, dict) and "score" in score_field and isinstance(score_field["score"], (float,int)):
        return float(score_field["score"])
    return 1.0

def draw_mask_from_polys(mask: np.ndarray, polys: List[np.ndarray]) -> np.ndarray:
    for pts in polys:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask

def poly_properties(pts: np.ndarray):
    contour = pts.astype(np.float32)
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
    x,y,w,h = cv2.boundingRect(contour.astype(np.int32))
    ar = float(w) / float(h) if h > 0 else 0.0
    hull = cv2.convexHull(contour)
    sol = (float(cv2.contourArea(contour)) / float(cv2.contourArea(hull))) if cv2.contourArea(hull) > 0 else 0.0
    return area, circ, ar, sol

def is_roundish(pts: np.ndarray, circ_thr=0.78, ar_min=0.75, ar_max=1.33, solidity_thr=0.95) -> bool:
    area, circ, ar, sol = poly_properties(pts)
    return (area > 0.0) and (circ >= circ_thr) and (ar_min <= ar <= ar_max) and (sol >= solidity_thr)

def rec_confidence_on_crop(ocr, image_bgr: np.ndarray) -> float:
    try:
        out = ocr_call(ocr, image_bgr, det=False, rec=True)
    except Exception:
        return 0.0
    best = 0.0
    if out and len(out) > 0 and out[0]:
        for item in out[0]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text, conf = item[0], item[1]
                try: conf = float(conf)
                except Exception: conf = 0.0
                if isinstance(text, str) and len(text.strip()) >= 1:
                    best = max(best, conf)
    return float(best)

def _atomic_write_csv(df: pd.DataFrame, path: Path, delimiter: str = ","):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, sep=delimiter, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(path)

def stage_detect(
    metadata_csv: Path,
    repo_root: Path,
    out_root: Path,
    min_conf: float = 0.35,
    min_area_px: int = 150,
    mask_dilate: int = 3,
    rotations: List[int] = [0],
    upscale_short_side: int = 1024,
    reject_round: bool = True,
    round_circ: float = 0.78,
    round_ar_min: float = 0.75,
    round_ar_max: float = 1.33,
    round_solidity: float = 0.95,
    rec_check_thresh: float = 0.0,
    delimiter: str = ",",
) -> None:
    df = pd.read_csv(metadata_csv, delimiter=delimiter, dtype=str, keep_default_na=False)
    for c in ["has_ocr","ocr_mask_path","ocr_boxes_path"]:
        if c not in df.columns: df[c] = ""

    ocr = init_paddle_ocr()
    total, hits, neg = len(df), 0, 0

    for idx in tqdm(range(total), desc="OCR detect"):
        row = df.iloc[idx]
        if str(row.get("is_nsfw","")).strip().lower() in {"1","true"}:
            continue

        img_rel = row.get("path_raw","").strip()
        if not img_rel:
            if row.get("has_ocr","") == "": df.at[idx,"has_ocr"]="False"; neg+=1
            continue

        img_path = (repo_root / img_rel) if (repo_root / img_rel).exists() else Path(img_rel)
        if not img_path.exists():
            if row.get("has_ocr","") == "": df.at[idx,"has_ocr"]="False"; neg+=1
            continue

        image, flattened_webp = load_image_with_webp_flatten(img_path)
        if image is None:
            if row.get("has_ocr","") == "": df.at[idx,"has_ocr"]="False"; neg+=1
            continue

        h, w = image.shape[:2]
        short = min(h, w)
        scale = 1.0
        image_for_ocr = image
        if upscale_short_side and short < upscale_short_side:
            scale = float(upscale_short_side) / float(short)
            image_for_ocr = cv2.resize(image, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_CUBIC)

        all_dets = []
        for a in rotations:
            a = int(a)
            if a % 360 == 0:
                im, invM = image_for_ocr, None
            else:
                hh, ww = image_for_ocr.shape[:2]
                R = cv2.getRotationMatrix2D((ww/2.0, hh/2.0), a, 1.0)
                invM = cv2.invertAffineTransform(R)
                im = cv2.warpAffine(image_for_ocr, R, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            try:
                rr = ocr_call(ocr, im, det=True, rec=False)
            except Exception:
                rr = None
            if rr and rr[0]:
                for det in rr[0]:
                    pts = np.array(det[0], dtype=np.float32)
                    if invM is not None: pts = cv2.transform(pts[None,:,:], invM)[0]
                    if scale != 1.0: pts *= (1.0/scale)
                    all_dets.append([pts, det[1] if len(det) > 1 else None])

        polys, boxes_dump = [], []
        for pts, score_field in all_dets:
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area < float(min_area_px): continue
            if reject_round and is_roundish(pts, round_circ, round_ar_min, round_ar_max, round_solidity): continue
            conf = parse_conf(score_field)
            if conf < float(min_conf): continue
            if rec_check_thresh > 0.0:
                x,y,w_box,h_box = cv2.boundingRect(pts.astype(np.int32))
                crop = image[max(0,y-2):min(h,y+h_box+2), max(0,x-2):min(w,x+w_box+2)].copy()
                if rec_confidence_on_crop(ocr, crop) < float(rec_check_thresh): continue
            polys.append(pts)
            boxes_dump.append({"points": pts.tolist(), "conf": float(conf), "area": area})

        has_ocr = len(polys) > 0
        band = (row.get("band","bandX") or "bandX").strip()
        species = (row.get("species","Unknown") or "Unknown").strip()
        image_id = row.get("image_id", img_path.stem)

        if has_ocr:
            hits += 1
            out_dir = out_root / band / species
            ensure_dir(out_dir)
            ext = ".png" if (img_path.suffix.lower()==".webp" and flattened_webp) else img_path.suffix.lower()
            copy_path = out_dir / f"{image_id}{ext}"
            cv2.imwrite(str(copy_path), image)

            mask = draw_mask_from_polys(np.zeros((h,w), np.uint8), polys)
            if mask_dilate and int(mask_dilate) > 0:
                k = int(mask_dilate)
                mask = cv2.dilate(mask, np.ones((k,k), np.uint8), 1)
            mask_path = out_dir / f"{image_id}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            boxes_path = out_dir / f"{image_id}_boxes.json"
            with open(boxes_path, "w", encoding="utf-8") as f:
                json.dump({"image": str(img_path), "width": w, "height": h, "boxes": boxes_dump}, f, ensure_ascii=False, indent=2)

            df.at[idx,"has_ocr"]="True"
            df.at[idx,"ocr_mask_path"]=str(mask_path)
            df.at[idx,"ocr_boxes_path"]=str(boxes_path)
        else:
            if df.at[idx,"has_ocr"] == "":
                df.at[idx,"has_ocr"]="False"; neg+=1

        if (idx % 200)==0:
            pass

    _atomic_write_csv(df, metadata_csv, delimiter)
    print(f"[detect] positives={hits}, negatives_marked={neg}, total={total}")

# ---- LaMa inpainting (with tiling) ----

def load_lama_model(config_path: Path, checkpoint_path: Path, device: str = "cuda"):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(config_path))
    model = load_checkpoint(cfg, str(checkpoint_path), strict=False, map_location=device)
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def _inpaint_core(model, img_bgr: np.ndarray, mask_gray: np.ndarray, device: str):
    import torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    mask_bin = (mask_gray > 127).astype(np.float32)
    img_t  = torch.from_numpy(img_rgb).permute(2,0,1)[None].contiguous()
    mask_t = torch.from_numpy(mask_bin)[None,None].contiguous()
    batch = move_to_device({"image": img_t, "mask": mask_t}, device)
    with torch.no_grad():
        out = model(batch)
    pred = out.get("inpainted", None) if isinstance(out, dict) else out
    pred = (pred[0].detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()*255.0+0.5).astype(np.uint8)
    return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

def _tiled_inpaint(model, img_bgr: np.ndarray, mask_gray: np.ndarray, device: str, tile: int = 1024, overlap: int = 64) -> np.ndarray:
    H,W = img_bgr.shape[:2]
    stride = max(1, tile - overlap)
    out = np.zeros_like(img_bgr, np.float32)
    acc = np.zeros((H,W), np.float32)

    win = np.ones((tile,tile), np.float32)
    e = overlap//2
    if e>0:
        ramp = np.linspace(0,1,e,dtype=np.float32)
        win[:e,:] *= ramp[:,None]; win[-e:,:] *= ramp[::-1][:,None]
        win[:, :e] *= ramp[None,:]; win[:, -e:] *= ramp[None,::-1]
    win3 = np.dstack([win,win,win])

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1, x1 = min(y0+tile,H), min(x0+tile,W)
            ph, pw = y0+tile-y1, x0+tile-x1
            img_t = img_bgr[y0:y1, x0:x1]
            msk_t = mask_gray[y0:y1, x0:x1]
            if ph or pw:
                img_t = cv2.copyMakeBorder(img_t,0,ph,0,pw,cv2.BORDER_REFLECT_101)
                msk_t = cv2.copyMakeBorder(msk_t,0,ph,0,pw,cv2.BORDER_CONSTANT,value=0)
            out_t = _inpaint_core(model, img_t, msk_t, device)
            if ph or pw:
                out_t = out_t[:tile-ph, :tile-pw]
                win_use  = win[:tile-ph, :tile-pw]
                win3_use = win3[:tile-ph,:tile-pw]
            else:
                win_use, win3_use = win, win3
            oy, ox = y0+out_t.shape[0], x0+out_t.shape[1]
            region = out[y0:oy, x0:ox]
            region[:] = region*(1.0-win3_use) + out_t.astype(np.float32)*win3_use
            acc[y0:oy, x0:ox] += win_use

    nz = acc > 1e-6
    out[nz] = (out[nz]/acc[nz,None]).clip(0,255)
    return out.astype(np.uint8)

def inpaint_one(model, img_bgr: np.ndarray, mask_gray: np.ndarray, device: str = "cuda") -> np.ndarray:
    hi, wi = img_bgr.shape[:2]
    hm, wm = mask_gray.shape[:2]
    if (hi,wi)!=(hm,wm):
        if hm<hi or wm<wi:
            mask_gray = cv2.copyMakeBorder(mask_gray,0,max(hi-hm,0),0,max(wi-wm,0),cv2.BORDER_CONSTANT,value=0)
        if hm>hi or wm>wi:
            mask_gray = mask_gray[:hi,:wi]
    if (hi*wi)/1e6 > 8.0:
        return _tiled_inpaint(model, img_bgr, mask_gray, device, tile=1024, overlap=64)
    def _pad8(arr, is_mask):
        h,w = arr.shape[:2]
        ph, pw = (8-h%8)%8, (8-w%8)%8
        if ph or pw:
            btype = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
            arr = cv2.copyMakeBorder(arr,0,ph,0,pw,btype,value=0)
        return arr, ph, pw
    img_p, phi, pwi = _pad8(img_bgr, False)
    msk_p, phm, pwm = _pad8(mask_gray, True)
    ph, pw = max(phi,phm), max(pwi,pwm)
    if ph or pw:
        if phi!=ph or pwi!=pw:
            img_p = cv2.copyMakeBorder(img_p,0,ph-phi,0,pw-pwi,cv2.BORDER_REFLECT_101)
        if phm!=ph or pwm!=pw:
            msk_p = cv2.copyMakeBorder(msk_p,0,ph-phm,0,pw-pwm,cv2.BORDER_CONSTANT,value=0)
    out = _inpaint_core(model, img_p, msk_p, device)
    if ph or pw: out = out[:hi,:wi]
    return out

def stage_inpaint(input_root: Path, output_root: Path, device: str, mask_suffix: str, lama_config: Path, lama_checkpoint: Path) -> None:
    model = load_lama_model(lama_config, lama_checkpoint, device=device)
    processed = 0
    for mask_path in tqdm(list(input_root.rglob(f"*{mask_suffix}")), desc="LaMa inpaint"):
        sp_dir = mask_path.parent
        stem = mask_path.name[: -len(mask_suffix)]
        img_path = None
        for ext in (".png",".jpg",".jpeg",".webp",".bmp"):
            cand = sp_dir / f"{stem}{ext}"
            if cand.exists(): img_path = cand; break
        if img_path is None: continue
        out_dir = sp_dir if output_root.resolve()==input_root.resolve() else (output_root / sp_dir.relative_to(input_root))
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.png"
        if out_path.exists(): continue
        img_bgr, _ = load_image_with_webp_flatten(img_path)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gray is None: continue
        try:
            out_bgr = inpaint_one(model, img_bgr, mask_gray, device=device)
            cv2.imwrite(str(out_path), out_bgr)
            processed += 1
        except Exception as e:
            print(f"[inpaint] ERROR {img_path}: {e}")
    print(f"[inpaint] wrote={processed} -> {output_root}")

def main():
    ap = argparse.ArgumentParser(description="PaddleOCR detection + LaMa inpainting.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_det = sub.add_parser("detect")
    ap_det.add_argument("--metadata", type=Path, required=True)
    ap_det.add_argument("--repo-root", type=Path, required=True)
    ap_det.add_argument("--output-root", type=Path, required=True)
    ap_det.add_argument("--min-conf", type=float, default=0.35)
    ap_det.add_argument("--min-area-px", type=float, default=150)
    ap_det.add_argument("--mask-dilate", type=int, default=3)
    ap_det.add_argument("--rotate", type=str, default="0")
    ap_det.add_argument("--upscale-short-side", type=int, default=1024)
    ap_det.add_argument("--reject-round", type=lambda s: str(s).lower() not in {"0","false","no"}, default=True)
    ap_det.add_argument("--round-circ", type=float, default=0.78)
    ap_det.add_argument("--round-ar-min", type=float, default=0.75)
    ap_det.add_argument("--round-ar-max", type=float, default=1.33)
    ap_det.add_argument("--round-solidity", type=float, default=0.95)
    ap_det.add_argument("--rec-check-thresh", type=float, default=0.0)
    ap_det.add_argument("--delimiter", type=str, default=",")

    ap_inp = sub.add_parser("inpaint")
    ap_inp.add_argument("--input-root", type=Path, required=True)
    ap_inp.add_argument("--output-root", type=Path, required=True)
    ap_inp.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap_inp.add_argument("--mask-suffix", default="_mask.png")
    ap_inp.add_argument("--lama-config", type=Path, required=True)
    ap_inp.add_argument("--lama-checkpoint", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "detect":
        rotations = [int(s) for s in str(args.rotate).split(",") if s.strip()!=""]
        stage_detect(
            metadata_csv=args.metadata,
            repo_root=args.repo_root,
            out_root=args.output_root,
            min_conf=args.min_conf,
            min_area_px=args.min_area_px,
            mask_dilate=args.mask_dilate,
            rotations=rotations,
            upscale_short_side=args.upscale_short_side,
            reject_round=args.reject_round,
            round_circ=args.round_circ,
            round_ar_min=args.round_ar_min,
            round_ar_max=args.round_ar_max,
            round_solidity=args.round_solidity,
            rec_check_thresh=args.rec_check_thresh,
            delimiter=args.delimiter,
        )
    else:
        stage_inpaint(
            input_root=args.input_root,
            output_root=args.output_root,
            device=args.device,
            mask_suffix=args.mask_suffix,
            lama_config=args.lama_config,
            lama_checkpoint=args.lama_checkpoint,
        )

if __name__ == "__main__":
    main()

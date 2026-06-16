"""Overnight, unattended: download downtown Mapillary imagery, then (only if the CPU is
fast enough) extract per-image visual-feature fractions with a small segmentation model.

Everything is checkpointed and logged, so a sleep/crash loses at most one image. Outputs:
  data/raw/mapillary/images.jsonl       one row per image (id, lon, lat, date, angle, pano)
  data/raw/mapillary/img/<id>.jpg       downloaded thumbnails
  data/processed/visual_features.jsonl  per-image class fractions (if CV ran)
  data/processed/overnight_summary.json final summary
  data/processed/overnight.log          progress log

Run:  python scripts/overnight_imagery.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPH = "https://graph.mapillary.com"
ENUM_GRID = 12
ENUM_LIMIT = 2000
CV_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
CV_MAX_SECONDS = 10.0          # if slower than this per image on CPU, skip CV
CITYSCAPES = {0: "road", 1: "sidewalk", 2: "building", 5: "pole", 8: "vegetation", 10: "sky"}

RAW = ROOT / "data" / "raw" / "mapillary"
IMG = RAW / "img"
PROC = ROOT / "data" / "processed"
for d in (RAW, IMG, PROC):
    d.mkdir(parents=True, exist_ok=True)
LOG = PROC / "overnight.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def token() -> str:
    t = os.environ.get("MAPILLARY_TOKEN")
    if t:
        return t.strip()
    return (ROOT / "secrets" / "mapillary_token.txt").read_text().strip()


TOK = token()


def api(path: str, **params) -> dict:
    params["access_token"] = TOK
    url = f"{GRAPH}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                log(f"  api error (giving up): {e}")
                return {}
            time.sleep(3)
    return {}


def enumerate_images() -> dict:
    import yaml
    b = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
    dlon = (b["max_lon"] - b["min_lon"]) / ENUM_GRID
    dlat = (b["max_lat"] - b["min_lat"]) / ENUM_GRID
    out = {}
    capped = 0
    for r in range(ENUM_GRID):
        for c in range(ENUM_GRID):
            w = b["min_lon"] + c * dlon
            s = b["min_lat"] + r * dlat
            bbox = f"{w},{s},{w + dlon},{s + dlat}"
            data = api("images", bbox=bbox, limit=ENUM_LIMIT,
                       fields="id,captured_at,compass_angle,is_pano,geometry,thumb_1024_url"
                       ).get("data", [])
            if len(data) >= ENUM_LIMIT:
                capped += 1
            for im in data:
                coords = (im.get("geometry") or {}).get("coordinates") or [None, None]
                out[im["id"]] = {"id": im["id"], "lon": coords[0], "lat": coords[1],
                                 "captured_at": im.get("captured_at"),
                                 "compass_angle": im.get("compass_angle"),
                                 "is_pano": im.get("is_pano"),
                                 "thumb": im.get("thumb_1024_url")}
        log(f"  enumerated row {r + 1}/{ENUM_GRID}, unique so far: {len(out)}")
    if capped:
        log(f"  WARNING: {capped} cells hit the {ENUM_LIMIT} cap (slight undercount)")
    with (RAW / "images.jsonl").open("w", encoding="utf-8") as f:
        for rec in out.values():
            f.write(json.dumps({k: v for k, v in rec.items() if k != "thumb"}) + "\n")
    return out


def download(images: dict) -> int:
    done = 0
    for i, rec in enumerate(images.values()):
        p = IMG / f"{rec['id']}.jpg"
        if p.exists() and p.stat().st_size > 0:
            done += 1
            continue
        if not rec.get("thumb"):
            continue
        try:
            urllib.request.urlretrieve(rec["thumb"], str(p))
            done += 1
        except Exception as e:
            log(f"  download fail {rec['id']}: {e}")
        if (i + 1) % 200 == 0:
            log(f"  downloaded {done}/{len(images)}")
    return done


def run_cv() -> None:
    try:
        import torch  # noqa
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation  # noqa
    except Exception:
        log("CV: installing torch + transformers + pillow (one-time) ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch", "transformers", "pillow"], check=False)
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except Exception as e:
        log(f"CV: dependencies unavailable, skipping ({e}). Imagery is downloaded for a GPU pass.")
        return

    proc = AutoImageProcessor.from_pretrained(CV_MODEL, size={"height": 512, "width": 512})
    model = SegformerForSemanticSegmentation.from_pretrained(CV_MODEL).eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

    paths = sorted(IMG.glob("*.jpg"))
    if not paths:
        log("CV: no images to process.")
        return

    def infer(path):
        img = Image.open(path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        pred = logits.argmax(0).cpu().numpy()
        tot = pred.size
        return {name: round(float((pred == cid).sum()) / tot, 4) for cid, name in CITYSCAPES.items()}

    t0 = time.time()
    _ = infer(paths[0])
    per = time.time() - t0
    log(f"CV: benchmark {per:.1f}s/image on CPU ({len(paths)} images, est {per * len(paths) / 3600:.1f} h)")
    if per > CV_MAX_SECONDS:
        log(f"CV: too slow (> {CV_MAX_SECONDS}s/image). Leaving feature extraction for a GPU pass.")
        return

    out = PROC / "visual_features.jsonl"
    done_ids = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    with out.open("a", encoding="utf-8") as f:
        for i, path in enumerate(paths):
            iid = path.stem
            if iid in done_ids:
                continue
            try:
                feats = infer(path)
                f.write(json.dumps({"id": iid, **feats}) + "\n")
                f.flush()
            except Exception as e:
                log(f"  cv fail {iid}: {e}")
            if (i + 1) % 100 == 0:
                log(f"  cv {i + 1}/{len(paths)}")
    log("CV: done.")


def main():
    log("=== overnight job start ===")
    t0 = time.time()
    log("Stage 1: enumerating images ...")
    images = enumerate_images()
    log(f"Stage 1 done: {len(images)} unique images.")
    log("Stage 2: downloading thumbnails ...")
    n = download(images)
    log(f"Stage 2 done: {n} images on disk.")
    log("Stage 3: CV feature extraction (gated) ...")
    try:
        run_cv()
    except Exception as e:
        log(f"CV stage error: {e}")
    summary = {
        "images_enumerated": len(images),
        "images_downloaded": len(list(IMG.glob('*.jpg'))),
        "visual_features_rows": sum(1 for _ in (PROC / "visual_features.jsonl").open())
        if (PROC / "visual_features.jsonl").exists() else 0,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (PROC / "overnight_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"=== overnight job done: {json.dumps(summary)} ===")


if __name__ == "__main__":
    main()

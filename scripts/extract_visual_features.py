"""Extract the B-layer: per-segment visual features from Mapillary imagery.

For each walk segment, pick up to K leaf-on images, run SegFormer-b0 (Cityscapes), and
record mean class fractions (road, sidewalk, building, pole, vegetation, sky). Checkpointed
per image, so a stop/crash resumes cleanly. No labels needed.

Outputs to data/processed/:
  visual_features.jsonl          per-image class fractions (checkpoint)
  segment_visual_features.csv    per-segment aggregated features
  extract.log                    progress

Run:  python scripts/extract_visual_features.py
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import box  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
METRIC = "EPSG:32611"
IMGDIR = ROOT / "data" / "raw" / "mapillary" / "img"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
K = 2
MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
CLASSES = {0: "road", 1: "sidewalk", 2: "building", 5: "pole", 8: "vegetation", 10: "sky"}
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}


def log(m):
    print(m, flush=True)
    with (OUT / "extract.log").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def coerce(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return "" if v is None else str(v)


def segments():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    E["seg_id"] = np.arange(len(E))
    return E.to_crs(METRIC)[["seg_id", "geometry"]]


def select_images(segs):
    rows = [json.loads(x) for x in (ROOT / "data" / "raw" / "mapillary" / "images.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                         crs="EPSG:4326").to_crs(METRIC)
    j = gpd.sjoin_nearest(g, segs, max_distance=20, distance_col="d")
    j["month"] = pd.to_datetime(j["captured_at"], unit="ms", errors="coerce").dt.month
    j["leaf_on"] = j["month"].between(4, 10).fillna(False)
    j["on_disk"] = j["id"].map(lambda i: (IMGDIR / f"{i}.jpg").exists())
    j = j[j["on_disk"]]
    j = j.sort_values(["seg_id", "leaf_on", "d"], ascending=[True, False, True])
    picked = j.groupby("seg_id").head(K)
    seg_of = dict(zip(picked["id"].astype(str), picked["seg_id"]))
    return seg_of


def run_cv(image_ids):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    proc = AutoImageProcessor.from_pretrained(MODEL, size={"height": 512, "width": 512})
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    feat_path = OUT / "visual_features.jsonl"
    done = set()
    if feat_path.exists():
        for line in feat_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [i for i in image_ids if i not in done]
    log(f"CV: {len(todo)} images to process ({len(done)} already done)")
    t0 = time.time()
    with feat_path.open("a", encoding="utf-8") as f:
        for n, iid in enumerate(todo):
            try:
                img = Image.open(IMGDIR / f"{iid}.jpg").convert("RGB")
                inp = proc(images=img, return_tensors="pt")
                with torch.no_grad():
                    pred = model(**inp).logits[0].argmax(0).cpu().numpy()
                tot = pred.size
                rec = {"id": iid, **{CLASSES[c]: round(float((pred == c).sum()) / tot, 4)
                                     for c in CLASSES}}
                f.write(json.dumps(rec) + "\n"); f.flush()
            except Exception as e:
                log(f"  fail {iid}: {e}")
            if (n + 1) % 500 == 0:
                rate = (time.time() - t0) / (n + 1)
                log(f"  {n + 1}/{len(todo)}  {rate:.2f}s/img  eta {rate * (len(todo) - n - 1) / 60:.0f} min")


def aggregate(seg_of):
    feat_path = OUT / "visual_features.jsonl"
    feats = [json.loads(x) for x in feat_path.read_text(encoding="utf-8").splitlines()]
    fd = pd.DataFrame(feats)
    fd["seg_id"] = fd["id"].map(seg_of)
    fd = fd.dropna(subset=["seg_id"])
    agg = fd.groupby("seg_id")[list(CLASSES.values())].mean().round(4)
    agg["n_img_used"] = fd.groupby("seg_id").size()
    agg = agg.reset_index()
    agg["seg_id"] = agg["seg_id"].astype(int)
    agg.to_csv(OUT / "segment_visual_features.csv", index=False)
    log(f"aggregated {len(agg)} segments -> segment_visual_features.csv")
    log("mean per-segment fractions: " + json.dumps(
        {c: round(float(agg[c].mean()), 3) for c in CLASSES.values()}))


def main():
    log("=== extract B-layer start ===")
    segs = segments()
    log(f"segments: {len(segs)}")
    seg_of = select_images(segs)
    log(f"selected images (<= {K}/segment, leaf-on preferred, on disk): {len(seg_of)}")
    run_cv(list(seg_of.keys()))
    aggregate(seg_of)
    log("=== extract B-layer done ===")


if __name__ == "__main__":
    main()

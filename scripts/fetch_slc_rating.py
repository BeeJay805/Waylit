"""Build a 3rd-city (SLC) blind-rating set: download daytime Mapillary photos for SLC walk
segments stratified by business density (SLC has NO local streetlight inventory, so no lighting
feature), then seed pairs. Same method as fetch_la_rating.py. Time-boxed defaults for an overnight
expansion run. Token from MAPILLARY_TOKEN or secrets/mapillary_token.txt.

Outputs: data/raw/mapillary_slc/img/<id>.jpg, data/processed/slc_segment_image_link.parquet,
         data/ground_truth/slc_pairwise_queue.csv
Run: python scripts/fetch_slc_rating.py
"""
import argparse
import json
import math
import os
import pathlib
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
CITY = "slc"
IMG = ROOT / "data" / "raw" / "mapillary_slc" / "img"   # reassigned per --city in main()
GRAPH = "https://graph.mapillary.com"
N_TARGET = 110          # covered segments (time-boxed)
MAX_QUERIES = 700
R = 0.00055
SEED = 11
N_PAIRS = {"strong": 40, "mild": 30, "near": 30}
CAP = 4


def token():
    t = os.environ.get("MAPILLARY_TOKEN")
    return t.strip() if t else (ROOT / "secrets" / "mapillary_token.txt").read_text().strip()


TOK = token()


def api_images(bbox):
    url = f"{GRAPH}/images?" + urllib.parse.urlencode(
        {"access_token": TOK, "bbox": bbox, "limit": 10,
         "fields": "id,geometry,is_pano,thumb_1024_url"})
    for a in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r).get("data", [])
        except Exception:
            if a == 2:
                return None
            time.sleep(2)
    return None


def nearest_photo(lon, lat):
    d = api_images(f"{lon - R},{lat - R},{lon + R},{lat + R}")
    if not d:
        return None
    best, bd = None, 1e9
    for im in d:
        if im.get("is_pano") or not im.get("geometry") or not im.get("thumb_1024_url"):
            continue
        x, y = im["geometry"]["coordinates"]
        dist = math.hypot((x - lon) * 111000 * math.cos(math.radians(lat)), (y - lat) * 111000)
        if dist < bd:
            best, bd = im, dist
    return best


def collect(df):
    IMG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["bt"] = pd.qcut(df.poi_density.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    buckets = {}
    for bt, grp in df.groupby("bt"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        buckets[bt] = list(idx)
    keys = list(buckets)
    kept, queries = [], 0
    while len(kept) < N_TARGET and queries < MAX_QUERIES:
        progressed = False
        for k in keys:
            if not buckets[k] or len(kept) >= N_TARGET:
                continue
            i = buckets[k].pop()
            progressed = True
            queries += 1
            s = df.loc[i]
            im = nearest_photo(float(s.lon), float(s.lat))
            if im is None:
                continue
            iid = str(im["id"])
            p = IMG / f"{iid}.jpg"
            if not p.exists():
                try:
                    urllib.request.urlretrieve(im["thumb_1024_url"], str(p))
                except Exception:
                    continue
            kept.append({"seg_id": int(s.seg_id), "img_best": iid, "lon": float(s.lon),
                         "lat": float(s.lat), "poi_density": int(s.poi_density),
                         "encl_height": float(s.encl_height), "encl_frontage": float(s.encl_frontage)})
            if len(kept) % 20 == 0:
                print(f"  collected {len(kept)}/{N_TARGET} (queried {queries})", flush=True)
        if not progressed:
            break
    print(f"collected {len(kept)} covered segments from {queries} queries", flush=True)
    return pd.DataFrame(kept)


def seed_pairs(link):
    rng = np.random.default_rng(SEED)
    d = link.reset_index(drop=True)
    for c in ["poi_density", "encl_height"]:
        v = d[c].to_numpy(dtype=float)
        d["z_" + c] = (v - v.mean()) / (v.std() + 1e-9)
    GRID = 4
    for ax, col in (("lon", "cx"), ("lat", "cy")):
        e = np.quantile(d[ax], np.linspace(0, 1, GRID + 1))
        d[col] = np.clip(np.digitize(d[ax], e[1:-1]), 0, GRID - 1)
    d["cell"] = d["cx"] * GRID + d["cy"]
    seg = d["seg_id"].to_numpy()
    zl, zp = d["z_poi_density"].to_numpy(), d["z_encl_height"].to_numpy()
    n = len(d)
    a, b = rng.integers(0, n, 40000), rng.integers(0, n, 40000)
    ok = a != b
    ds = np.hypot(zl[a[ok]] - zl[b[ok]], zp[a[ok]] - zp[b[ok]])
    p50, p80 = np.quantile(ds, [0.5, 0.8])
    bucket = lambda x: "strong" if x > p80 else ("mild" if x > p50 else "near")
    counts = {k: 0 for k in N_PAIRS}
    appear = np.zeros(n, dtype=int)
    used, pairs, att = set(), [], 0
    while sum(counts.values()) < sum(N_PAIRS.values()) and att < 400000:
        att += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j or appear[i] >= CAP or appear[j] >= CAP:
            continue
        key = (min(seg[i], seg[j]), max(seg[i], seg[j]))
        if key in used:
            continue
        dist = float(np.hypot(zl[i] - zl[j], zp[i] - zp[j]))
        bk = bucket(dist)
        if counts[bk] >= N_PAIRS[bk]:
            continue
        used.add(key); counts[bk] += 1; appear[i] += 1; appear[j] += 1
        ra, rb = d.iloc[i], d.iloc[j]
        pairs.append({"pair_id": len(pairs), "seg_a": int(ra.seg_id), "seg_b": int(rb.seg_id),
                      "img_a": ra.img_best, "img_b": rb.img_best,
                      "lat_a": round(float(ra.lat), 6), "lon_a": round(float(ra.lon), 6),
                      "lat_b": round(float(rb.lat), 6), "lon_b": round(float(rb.lon), 6),
                      "pair_type": bk, "poi_a": int(ra.poi_density), "poi_b": int(rb.poi_density),
                      "cell_a": int(ra.cell), "cell_b": int(rb.cell)})
    q = pd.DataFrame(pairs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    q["pair_id"] = np.arange(len(q))
    q.to_csv(GT / f"{CITY}_pairwise_queue.csv", index=False)
    print(f"seeded {len(q)} pairs | types {dict(q.pair_type.value_counts())} | distinct segs "
          f"{len(pd.unique(q[['seg_a','seg_b']].to_numpy().ravel()))}", flush=True)


def main():
    global CITY, IMG
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="slc")
    CITY = ap.parse_args().city
    IMG = ROOT / "data" / "raw" / f"mapillary_{CITY}" / "img"
    df = pd.read_parquet(PROC / f"{CITY}_features.parquet")
    link_path = PROC / f"{CITY}_segment_image_link.parquet"
    if link_path.exists():
        link = pd.read_parquet(link_path)
        print(f"reusing {len(link)} cached covered segments")
    else:
        link = collect(df)
        link.to_parquet(link_path)
    if len(link) < 20:
        print(f"ONLY {len(link)} covered segments - too few; SLC Mapillary coverage too sparse.")
        return
    seed_pairs(link)
    print("SLC rating set ready.")


if __name__ == "__main__":
    main()

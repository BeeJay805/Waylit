"""Build the LA blind-rating set: download daytime Mapillary photos for ~300 transect segments
that span the lighting/business range, then seed 150 pairs. LA Mapillary coverage is ~37% at
the segment level, so we query candidates round-robin across lighting x business strata and keep
the covered ones, which preserves variance despite the coverage skew toward arterials.

NO model scores are shown by the rating tool; raw features are kept only for later analysis.
Reads data/processed/la_seg_features.parquet (from validate_city). Token from MAPILLARY_TOKEN
or secrets/mapillary_token.txt.

Outputs:
  data/raw/mapillary_la/img/<id>.jpg
  data/processed/la_segment_image_link.parquet
  data/ground_truth/la_pairwise_queue.csv
  data/processed/la_locator_base.png
Run:  python scripts/fetch_la_rating.py
"""
import json
import math
import os
import pathlib
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
IMG = ROOT / "data" / "raw" / "mapillary_la" / "img"
GRAPH = "https://graph.mapillary.com"
N_TARGET = 300          # covered segments to collect
MAX_QUERIES = 1400      # API budget to find them
R = 0.00055             # ~55 m half-box around a segment
SEED = 7
N_PAIRS = {"strong": 60, "mild": 45, "near": 45}
CAP = 4


def token():
    t = os.environ.get("MAPILLARY_TOKEN")
    return t.strip() if t else (ROOT / "secrets" / "mapillary_token.txt").read_text().strip()


TOK = token()


def api_images(bbox):
    url = f"{GRAPH}/images?" + urllib.parse.urlencode(
        {"access_token": TOK, "bbox": bbox, "limit": 10,
         "fields": "id,geometry,is_pano,captured_at,thumb_1024_url"})
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
    """Round-robin over lighting x business strata; keep covered segments, download a photo each."""
    IMG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["lt"] = pd.qcut(df.lit_frac.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    df["bt"] = pd.qcut(df.poi_density.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    buckets = {}
    for (lt, bt), grp in df.groupby(["lt", "bt"]):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        buckets[(lt, bt)] = list(idx)
    keys = list(buckets)
    per_cap = math.ceil(N_TARGET / len(keys)) + 4
    kept, used, queries = [], 0, 0
    cnt = {k: 0 for k in keys}
    while len(kept) < N_TARGET and queries < MAX_QUERIES:
        progressed = False
        for k in keys:
            if not buckets[k] or cnt[k] >= per_cap or len(kept) >= N_TARGET:
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
            cnt[k] += 1
            kept.append({"seg_id": int(s.seg_id), "img_best": iid,
                         "lon": float(s.lon), "lat": float(s.lat),
                         "lit_frac": float(s.lit_frac), "poi_density": int(s.poi_density),
                         "income": float(s.income) if pd.notna(s.income) else np.nan,
                         "pct_nonwhite": float(s.pct_nonwhite) if pd.notna(s.pct_nonwhite) else np.nan})
            if len(kept) % 25 == 0:
                print(f"  collected {len(kept)}/{N_TARGET} (queried {queries})", flush=True)
        if not progressed:
            break
    print(f"collected {len(kept)} covered segments from {queries} queries", flush=True)
    return pd.DataFrame(kept)


def seed_pairs(link):
    rng = np.random.default_rng(SEED)
    d = link.reset_index(drop=True)
    for c in ["lit_frac", "poi_density"]:
        v = d[c].to_numpy(dtype=float)
        d["z_" + c] = (v - v.mean()) / (v.std() + 1e-9)
    GRID = 4
    for ax, col in (("lon", "cx"), ("lat", "cy")):
        e = np.quantile(d[ax], np.linspace(0, 1, GRID + 1))
        d[col] = np.clip(np.digitize(d[ax], e[1:-1]), 0, GRID - 1)
    d["cell"] = d["cx"] * GRID + d["cy"]
    seg, zl, zp = d["seg_id"].to_numpy(), d["z_lit_frac"].to_numpy(), d["z_poi_density"].to_numpy()
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
                      "pair_type": bk, "lit_a": round(float(ra.lit_frac), 3),
                      "lit_b": round(float(rb.lit_frac), 3), "poi_a": int(ra.poi_density),
                      "poi_b": int(rb.poi_density), "cell_a": int(ra.cell), "cell_b": int(rb.cell)})
    q = pd.DataFrame(pairs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    q["pair_id"] = np.arange(len(q))
    GT.mkdir(parents=True, exist_ok=True)
    q.to_csv(GT / "la_pairwise_queue.csv", index=False)
    cells = pd.unique(q[["cell_a", "cell_b"]].to_numpy().ravel())
    print(f"seeded {len(q)} pairs | types {dict(q.pair_type.value_counts())} | "
          f"cells {len(cells)}/{GRID*GRID} | distinct segs "
          f"{len(pd.unique(q[['seg_a','seg_b']].to_numpy().ravel()))}", flush=True)


def render_locator(B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import osmnx as ox
    from shapely.geometry import box
    G = ox.graph_from_polygon(box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"]),
                              network_type="walk", retain_all=True)
    E = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")
    mean_lat = math.radians((B["min_lat"] + B["max_lat"]) / 2)
    w = 540
    h = int(w * (B["max_lat"] - B["min_lat"]) / ((B["max_lon"] - B["min_lon"]) * math.cos(mean_lat)))
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    E.plot(ax=ax, color="#c9ccd1", linewidth=0.4)
    ax.set_xlim(B["min_lon"], B["max_lon"]); ax.set_ylim(B["min_lat"], B["max_lat"])
    fig.savefig(PROC / "la_locator_base.png", dpi=100)
    plt.close(fig)
    print("rendered la_locator_base.png", flush=True)


def main():
    cfg = yaml.safe_load((ROOT / "config" / "cities" / "la.yaml").read_text())
    B = cfg["bbox"]
    df = pd.read_parquet(PROC / "la_seg_features.parquet")
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.mx, df.my),
                         crs=cfg["crs_metric"]).to_crs("EPSG:4326")
    df["lon"], df["lat"] = g.geometry.x.values, g.geometry.y.values

    link_path = PROC / "la_segment_image_link.parquet"
    if link_path.exists():
        link = pd.read_parquet(link_path)
        print(f"reusing {len(link)} cached covered segments (delete {link_path.name} to refetch)")
    else:
        link = collect(df)
        link.to_parquet(link_path)
    cov = link[["lit_frac", "poi_density", "income"]]
    print(f"covered-set variance: lit_frac {cov.lit_frac.min():.2f}-{cov.lit_frac.max():.2f} "
          f"med {cov.lit_frac.median():.2f} | poi med {int(cov.poi_density.median())} "
          f"p90 {int(cov.poi_density.quantile(.9))} | income "
          f"${np.nanpercentile(cov.income,5):,.0f}-${np.nanpercentile(cov.income,95):,.0f}")
    seed_pairs(link)
    render_locator(B)
    print("LA rating set ready. Run:  python scripts/blind_compare.py --city la --host 0.0.0.0")


if __name__ == "__main__":
    main()

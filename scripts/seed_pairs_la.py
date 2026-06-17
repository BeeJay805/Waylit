"""Seed a disjoint batch of blind LA pairwise comparisons (same method as seed_pairs.py but
on the LA image-backed segments). Difficulty mixed on a standardized (lit_frac, poi_density)
space; per-segment appearance cap keeps it broad; spatial cells spread it; pairs already in an
--exclude queue are avoided. Only segments whose Mapillary image file exists are used (so both
sides are showable). NO scores shown by the rating tool; deterministic.

Run: python scripts/seed_pairs_la.py --seed 99 --out la_pairwise_queue2.csv --exclude la_pairwise_queue.csv
"""
import argparse
import csv
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
IMGDIR = ROOT / "data" / "raw" / "mapillary_la" / "img"
N_TARGET = {"strong": 60, "mild": 45, "near": 45}
CAP = 4
GRID = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="la_pairwise_queue2.csv")
    ap.add_argument("--exclude", default="la_pairwise_queue.csv")
    args = ap.parse_args()

    link = pd.read_parquet(PROC / "la_segment_image_link.parquet")
    d = link[["seg_id", "img_best", "lat", "lon", "lit_frac", "poi_density"]].dropna(
        subset=["img_best", "lit_frac", "poi_density"]).copy()
    d["have"] = d["img_best"].astype(str).map(lambda x: (IMGDIR / f"{x}.jpg").exists())
    d = d[d["have"]].reset_index(drop=True)
    print(f"LA segments with present image: {len(d)}")

    for c in ("lit_frac", "poi_density"):
        s = d[c].to_numpy(dtype=float)
        d["z_" + c] = (s - s.mean()) / (s.std() + 1e-9)
    for ax, col in (("lon", "cx"), ("lat", "cy")):
        v = d[ax].to_numpy()
        edges = np.quantile(v, np.linspace(0, 1, GRID + 1))
        d[col] = np.clip(np.digitize(v, edges[1:-1]), 0, GRID - 1)
    d["cell"] = d["cx"] * GRID + d["cy"]

    rng = np.random.default_rng(args.seed)
    seg = d["seg_id"].to_numpy()
    zl, zp = d["z_lit_frac"].to_numpy(), d["z_poi_density"].to_numpy()
    n = len(d)

    a = rng.integers(0, n, 40000)
    b = rng.integers(0, n, 40000)
    ok = a != b
    dist_s = np.hypot(zl[a[ok]] - zl[b[ok]], zp[a[ok]] - zp[b[ok]])
    p50, p80 = np.quantile(dist_s, [0.50, 0.80])

    def bucket(dist):
        return "strong" if dist > p80 else "mild" if dist > p50 else "near"

    used = set()
    for r in csv.DictReader((GT / args.exclude).open(encoding="utf-8")):
        x, y = int(r["seg_a"]), int(r["seg_b"])
        used.add((min(x, y), max(x, y)))
    print(f"excluding {len(used)} existing pairs from {args.exclude}")

    counts = {k: 0 for k in N_TARGET}
    appear = np.zeros(n, dtype=int)
    pairs = []
    attempts = 0
    while sum(counts.values()) < sum(N_TARGET.values()) and attempts < 400000:
        attempts += 1
        i, jx = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == jx or appear[i] >= CAP or appear[jx] >= CAP:
            continue
        key = (min(seg[i], seg[jx]), max(seg[i], seg[jx]))
        if key in used:
            continue
        dist = float(np.hypot(zl[i] - zl[jx], zp[i] - zp[jx]))
        bk = bucket(dist)
        if counts[bk] >= N_TARGET[bk]:
            continue
        used.add(key)
        counts[bk] += 1
        appear[i] += 1
        appear[jx] += 1
        ra, rb = d.iloc[i], d.iloc[jx]
        pairs.append({
            "pair_id": len(pairs), "seg_a": int(ra.seg_id), "seg_b": int(rb.seg_id),
            "img_a": ra.img_best, "img_b": rb.img_best,
            "lat_a": round(float(ra.lat), 6), "lon_a": round(float(ra.lon), 6),
            "lat_b": round(float(rb.lat), 6), "lon_b": round(float(rb.lon), 6),
            "pair_type": bk, "pair_dist": round(dist, 3),
            "lit_a": round(float(ra.lit_frac), 3), "lit_b": round(float(rb.lit_frac), 3),
            "poi_a": int(ra.poi_density), "poi_b": int(rb.poi_density),
            "cell_a": int(ra.cell), "cell_b": int(rb.cell),
        })

    q = pd.DataFrame(pairs).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    q["pair_id"] = np.arange(len(q))
    q.to_csv(GT / args.out, index=False)
    print(f"pairs: {len(q)} | by type: {dict(q.pair_type.value_counts())} | "
          f"distinct segs: {len(pd.unique(q[['seg_a','seg_b']].to_numpy().ravel()))} | "
          f"cells: {len(pd.unique(q[['cell_a','cell_b']].to_numpy().ravel()))}/{GRID*GRID}")
    print(f"saved data/ground_truth/{args.out}")


if __name__ == "__main__":
    main()

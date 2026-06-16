"""Seed ~150 blind pairwise comparisons spanning lighting/business and downtown geography.

Pairs are drawn only from the 955 segments that have a daytime photo (so both sides are
showable and both have visual features for the B1/C models). Difficulty is mixed on purpose:
  strong  far apart in standardized (lighting, business) space  -> informative, easy calls
  mild    moderately apart
  near    close together                                        -> tests fine discrimination
A per-segment appearance cap keeps the set broad (no photo dominates), and uniform sampling
over the 955 (which already blanket the downtown bbox) spreads pairs geographically.

NO model scores are written for display; raw features are kept only for later analysis and
are never shown by the rating tool. Deterministic (fixed seed).

Output: data/ground_truth/pairwise_queue.csv
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
N_TARGET = {"strong": 60, "mild": 45, "near": 45}  # 150
CAP = 4          # max times a segment may appear
SEED = 42
GRID = 4         # spatial cells per axis, for spread reporting


def main():
    df = pd.read_parquet(PROC / "segment_features.parquet")
    link = pd.read_parquet(PROC / "segment_image_link.parquet")[["seg_id", "img_best"]]
    d = df[df.has_visual].merge(link, on="seg_id", how="inner").reset_index(drop=True)

    # standardized lighting/business; pair "distance" is euclidean in that 2-D space
    z = {}
    for c in ["light", "poi_density"]:
        s = d[c].to_numpy(dtype=float)
        z[c] = (s - s.mean()) / (s.std() + 1e-9)
    d["zl"], d["zp"] = z["light"], z["poi_density"]

    # spatial cell for spread reporting
    for ax, col in (("lon", "cx"), ("lat", "cy")):
        v = d[ax].to_numpy()
        edges = np.quantile(v, np.linspace(0, 1, GRID + 1))
        d[col] = np.clip(np.digitize(v, edges[1:-1]), 0, GRID - 1)
    d["cell"] = d["cx"] * GRID + d["cy"]

    rng = np.random.default_rng(SEED)
    seg = d["seg_id"].to_numpy()
    zl, zp = d["zl"].to_numpy(), d["zp"].to_numpy()
    n = len(d)

    # difficulty thresholds from a sample of random pair distances
    a = rng.integers(0, n, 40000)
    b = rng.integers(0, n, 40000)
    ok = a != b
    dist_s = np.hypot(zl[a[ok]] - zl[b[ok]], zp[a[ok]] - zp[b[ok]])
    p50, p80 = np.quantile(dist_s, [0.50, 0.80])

    def bucket(dist):
        if dist > p80:
            return "strong"
        if dist > p50:
            return "mild"
        return "near"

    counts = {k: 0 for k in N_TARGET}
    appear = np.zeros(n, dtype=int)
    used = set()
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
            "pair_id": len(pairs),
            "seg_a": int(ra.seg_id), "seg_b": int(rb.seg_id),
            "img_a": ra.img_best, "img_b": rb.img_best,
            "lat_a": round(float(ra.lat), 6), "lon_a": round(float(ra.lon), 6),
            "lat_b": round(float(rb.lat), 6), "lon_b": round(float(rb.lon), 6),
            "pair_type": bk, "pair_dist": round(dist, 3),
            "light_a": round(float(ra.light), 3), "light_b": round(float(rb.light), 3),
            "poi_a": int(ra.poi_density), "poi_b": int(rb.poi_density),
            "cell_a": int(ra.cell), "cell_b": int(rb.cell),
        })

    q = pd.DataFrame(pairs)
    # deterministic shuffle so the session is well mixed, then renumber
    q = q.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    q["pair_id"] = np.arange(len(q))

    GT.mkdir(parents=True, exist_ok=True)
    q.to_csv(GT / "pairwise_queue.csv", index=False)

    segs_used = pd.unique(q[["seg_a", "seg_b"]].to_numpy().ravel())
    cells = pd.unique(q[["cell_a", "cell_b"]].to_numpy().ravel())
    print(f"pairs: {len(q)} | by type: {dict(q.pair_type.value_counts())}")
    print(f"distinct segments used: {len(segs_used)} / {n} | max appearances: {appear.max()}")
    print(f"spatial cells touched: {len(cells)} / {GRID * GRID} | "
          f"dist thresholds p50={p50:.2f} p80={p80:.2f}")
    print("saved data/ground_truth/pairwise_queue.csv")


if __name__ == "__main__":
    main()

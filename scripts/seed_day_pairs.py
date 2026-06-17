"""Seed ~150 DAY-safety pairs for the blind rating tool.

Day walking safety is about traffic, not darkness, so pairs are drawn from the photographed
Boise segments and stratified across traffic exposure (car-free path .. busy arterial) and
crossing density, with geographic spread and a per-segment cap. Same photos as the night set;
the day MODE just asks a different question. NO scores are written for display.

Reads segment_features.parquet + walk_safety_features.parquet + segment_image_link.parquet.
Output: data/ground_truth/day_pairwise_queue.csv   (served by blind_compare.py --mode day)
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
N_TARGET = {"strong": 60, "mild": 45, "near": 45}
CAP = 4
SEED = 11
GRID = 4


def main():
    feat = pd.read_parquet(PROC / "segment_features.parquet")[["seg_id", "lat", "lon"]]
    day = pd.read_parquet(PROC / "walk_safety_features.parquet")
    link = pd.read_parquet(PROC / "segment_image_link.parquet")[["seg_id", "img_best"]]
    d = feat.merge(day, on="seg_id").merge(link, on="seg_id")
    d = d.dropna(subset=["traffic_exposure", "crossing_near", "car_free"]).reset_index(drop=True)
    print(f"photographed segments with day features: {len(d)} | "
          f"car-free {d.car_free.mean():.0%} | traffic_exposure med {d.traffic_exposure.median():.2f} "
          f"p90 {d.traffic_exposure.quantile(.9):.2f}")

    for c in ["traffic_exposure", "crossing_near"]:
        v = d[c].to_numpy(dtype=float)
        d["z_" + c] = (v - v.mean()) / (v.std() + 1e-9)
    for ax, col in (("lon", "cx"), ("lat", "cy")):
        e = np.quantile(d[ax], np.linspace(0, 1, GRID + 1))
        d[col] = np.clip(np.digitize(d[ax], e[1:-1]), 0, GRID - 1)
    d["cell"] = d["cx"] * GRID + d["cy"]

    rng = np.random.default_rng(SEED)
    seg = d["seg_id"].to_numpy()
    zt, zc = d["z_traffic_exposure"].to_numpy(), d["z_crossing_near"].to_numpy()
    n = len(d)
    a, b = rng.integers(0, n, 40000), rng.integers(0, n, 40000)
    ok = a != b
    ds = np.hypot(zt[a[ok]] - zt[b[ok]], zc[a[ok]] - zc[b[ok]])
    p50, p80 = np.quantile(ds, [0.5, 0.8])
    bucket = lambda x: "strong" if x > p80 else ("mild" if x > p50 else "near")

    counts = {k: 0 for k in N_TARGET}
    appear = np.zeros(n, dtype=int)
    used, pairs, att = set(), [], 0
    while sum(counts.values()) < sum(N_TARGET.values()) and att < 400000:
        att += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j or appear[i] >= CAP or appear[j] >= CAP:
            continue
        key = (min(seg[i], seg[j]), max(seg[i], seg[j]))
        if key in used:
            continue
        dist = float(np.hypot(zt[i] - zt[j], zc[i] - zc[j]))
        bk = bucket(dist)
        if counts[bk] >= N_TARGET[bk]:
            continue
        used.add(key); counts[bk] += 1; appear[i] += 1; appear[j] += 1
        ra, rb = d.iloc[i], d.iloc[j]
        pairs.append({
            "pair_id": len(pairs), "seg_a": int(ra.seg_id), "seg_b": int(rb.seg_id),
            "img_a": ra.img_best, "img_b": rb.img_best,
            "lat_a": round(float(ra.lat), 6), "lon_a": round(float(ra.lon), 6),
            "lat_b": round(float(rb.lat), 6), "lon_b": round(float(rb.lon), 6),
            "pair_type": bk,
            "traffic_a": round(float(ra.traffic_exposure), 3), "traffic_b": round(float(rb.traffic_exposure), 3),
            "carfree_a": int(ra.car_free), "carfree_b": int(rb.car_free),
            "cross_a": int(ra.crossing_near), "cross_b": int(rb.crossing_near),
            "cell_a": int(ra.cell), "cell_b": int(rb.cell)})

    q = pd.DataFrame(pairs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    q["pair_id"] = np.arange(len(q))
    GT.mkdir(parents=True, exist_ok=True)
    q.to_csv(GT / "day_pairwise_queue.csv", index=False)
    cells = pd.unique(q[["cell_a", "cell_b"]].to_numpy().ravel())
    segs = pd.unique(q[["seg_a", "seg_b"]].to_numpy().ravel())
    print(f"seeded {len(q)} pairs | types {dict(q.pair_type.value_counts())} | "
          f"distinct segs {len(segs)} | cells {len(cells)}/{GRID*GRID} | "
          f"car-free-vs-road pairs {int((q.carfree_a != q.carfree_b).sum())}")
    print("saved data/ground_truth/day_pairwise_queue.csv")


if __name__ == "__main__":
    main()

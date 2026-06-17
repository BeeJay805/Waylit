"""Fit the DAY-safety weights from the day blind-rating round (separate from the night score).

Daytime walking safety is about traffic and pedestrian infrastructure, so the features are the
day-safety layer (traffic exposure, crossings) plus the streetscape (enclosure, activity), NOT
the night features (lighting, night-transit). Transparent logistic (Bradley-Terry) on signed
feature differences; weights LEARNED, never hand-set. Daytime photos are the correct stimulus,
so no proxy gap. Crime/demographics are never inputs.

Reads data/ground_truth/day_pairwise.csv + segment_features.parquet + walk_safety_features.parquet.
Output: data/processed/day_fit.json
"""
import csv
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOG = ROOT / "data" / "ground_truth" / "day_pairwise.csv"
DAY = ["traffic_exposure", "crossing_near", "encl_frontage", "encl_height", "poi_density"]
LOG1P = {"crossing_near", "poi_density", "encl_height"}
CV_BLOCKS = (2, 3, 4, 5)
SEED = 11
MIN_PAIRS = 20


def load_Z():
    sf = pd.read_parquet(PROC / "segment_features.parquet")[
        ["seg_id", "lat", "lon", "encl_frontage", "encl_height", "poi_density"]]
    ws = pd.read_parquet(PROC / "walk_safety_features.parquet")[
        ["seg_id", "traffic_exposure", "crossing_near"]]
    df = sf.merge(ws, on="seg_id").dropna(subset=DAY).set_index("seg_id")
    Z = pd.DataFrame(index=df.index)
    for c in DAY:
        x = df[c].astype(float)
        if c in LOG1P:
            x = np.log1p(x.clip(lower=0))
        Z[c] = (x - x.mean()) / (x.std() + 1e-9)
    return Z, df[["lat", "lon"]]


def load_pairs():
    counts, pairs, raters = {"decisive": 0, "tie": 0, "skip": 0}, [], {}
    if not LOG.exists():
        return pairs, counts, raters
    with LOG.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ch = r.get("choice", "")
            raters[r.get("rater", "?")] = raters.get(r.get("rater", "?"), 0) + (ch in ("L", "R"))
            if ch in ("L", "R") and r.get("winner_seg") and r.get("loser_seg"):
                pairs.append((int(r["winner_seg"]), int(r["loser_seg"])))
                counts["decisive"] += 1
            elif ch in ("tie", "skip"):
                counts[ch] += 1
    return pairs, counts, raters


def diffs(pairs, Z):
    return Z.loc[[w for w, _ in pairs]].to_numpy() - Z.loc[[l for _, l in pairs]].to_numpy()


def fit_coef(pairs, Z, C=1.0):
    d = diffs(pairs, Z)
    X, y = np.vstack([d, -d]), np.r_[np.ones(len(d)), np.zeros(len(d))]
    return LogisticRegression(fit_intercept=False, C=C, max_iter=2000).fit(X, y).coef_[0]


def cv(pairs, Z, latlon, k):
    blk = dict(zip(latlon.index, KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(latlon.to_numpy())))
    correct, held = [], 0
    for f in range(k):
        test = [(w, l) for w, l in pairs if blk.get(w) == f and blk.get(l) == f]
        train = [(w, l) for w, l in pairs if blk.get(w) != f and blk.get(l) != f]
        if not test or len(train) < MIN_PAIRS:
            continue
        held += len(test)
        correct.extend((diffs(test, Z) @ fit_coef(train, Z) > 0).tolist())
    return (round(float(np.mean(correct)), 3) if correct else None), held


def single_feature_agreement(pairs, raw):
    out = {}
    for c in DAY:
        wv = np.array([raw.loc[w, c] for w, _ in pairs], dtype=float)
        lv = np.array([raw.loc[l, c] for _, l in pairs], dtype=float)
        m = wv != lv
        out[c] = {"agreement": round(float((wv > lv)[m].mean()), 3), "n_differ": int(m.sum())}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["agreement"]))


def main():
    Z, latlon = load_Z()
    raw = (pd.read_parquet(PROC / "segment_features.parquet")[["seg_id", "encl_frontage", "encl_height", "poi_density"]]
           .merge(pd.read_parquet(PROC / "walk_safety_features.parquet")[["seg_id", "traffic_exposure", "crossing_near"]],
                  on="seg_id").set_index("seg_id"))
    pairs, counts, raters = load_pairs()
    if len(pairs) < MIN_PAIRS:
        print(f"Only {len(pairs)} decisive DAY pairs (need >= {MIN_PAIRS}). Rate via "
              "python scripts/blind_compare.py --mode day")
        return

    coef = fit_coef(pairs, Z)
    rng = np.random.default_rng(SEED)
    boots = np.array([fit_coef([pairs[i] for i in rng.integers(0, len(pairs), len(pairs))], Z)
                      for _ in range(300)])
    weights = {c: {"coef": round(float(coef[i]), 3),
                   "ci95": [round(float(np.percentile(boots[:, i], 2.5)), 3),
                            round(float(np.percentile(boots[:, i], 97.5)), 3)]}
               for i, c in enumerate(DAY)}
    sfa = single_feature_agreement(pairs, raw)
    sweep = [{"blocks": k, **dict(zip(("agreement", "held_out"), cv(pairs, Z, latlon, k)))}
             for k in CV_BLOCKS]

    out = {"score": "DAY walking safety (separate from night comfort)", "rater_counts": raters,
           "n_decisive": len(pairs), "counts": counts, "features": DAY, "weights": weights,
           "robust_single_feature_agreement": sfa, "spatial_cv_sensitivity": sweep,
           "caveats": [
               "Single rater; weights learned not hand-set; crime/demographics never inputs.",
               "Daytime photos are the CORRECT stimulus for a day question (no night proxy gap).",
               "Photographed Boise segments are 80% car-free, so high-traffic arterials are under-"
               "represented; traffic_exposure variance in the rated set is modest.",
               "traffic_exposure expected NEGATIVE (more traffic = less safe); read signs accordingly.",
               "No bias audit here (Boise demographics not loaded); add a Census join to check."]}
    (PROC / "day_fit.json").write_text(json.dumps(out, indent=2))

    print(f"DAY fit | rater(s) {raters} | decisive {len(pairs)} (ties {counts['tie']})")
    print("\nlearned weights (standardized; +coef => safer-feeling by DAY):")
    for c in DAY:
        w = weights[c]
        sig = "" if (w["ci95"][0] <= 0 <= w["ci95"][1]) else "  *signif*"
        print(f"  {c:18s} {w['coef']:+.2f}  CI[{w['ci95'][0]:+.2f},{w['ci95'][1]:+.2f}]{sig}")
    print("\nrobust single-feature agreement (all decisive pairs):")
    for c, d in sfa.items():
        print(f"  pick higher {c:18s} {d['agreement']:.0%}  (n={d['n_differ']})")
    print("\nspatial-block CV (held-out agreement):")
    for s in sweep:
        print(f"  blocks {s['blocks']}  held {s['held_out']:>3}  agreement {s['agreement']}")
    print("wrote day_fit.json")


if __name__ == "__main__":
    main()

"""Fit comfort weights from the LA blind-rating round, then bias-audit the LEARNED weights.

Structured-only (no vision model was run on LA). This answers the question that sent us to a
high-variance, diverse city: do the rater's LA "feels safer" calls produce weights that track
income or race? We fit a transparent logistic (Bradley-Terry) on signed feature differences,
then apply the learned comfort score across LA and correlate it with ACS income and % nonwhite.
Crime/demographics are NEVER features; they are only the held-out audit targets.

Reads data/ground_truth/la_pairwise.csv + data/processed/la_seg_features.parquet.
Output: data/processed/la_fit.json
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
LOG = ROOT / "data" / "ground_truth" / "la_pairwise.csv"
FEATS = ["lit_frac", "encl_frontage", "encl_height", "poi_density", "poi_night_density"]
LOG1P = {"poi_density", "poi_night_density", "encl_height"}
CV_BLOCKS = (2, 3, 4, 5)
SEED = 7
MIN_PAIRS = 20


def load_Z():
    df = pd.read_parquet(PROC / "la_seg_features.parquet").set_index("seg_id")
    assert not any(c in ("crime",) for c in FEATS), "crime must never be a feature"
    Z = pd.DataFrame(index=df.index)
    for c in FEATS:
        x = df[c].astype(float)
        if c in LOG1P:
            x = np.log1p(x.clip(lower=0))
        Z[c] = (x - x.mean()) / (x.std() + 1e-9)
    Z = Z.dropna()
    meta = df.loc[Z.index, ["mx", "my", "income", "pct_nonwhite"]]
    return Z, meta


def load_pairs():
    counts = {"decisive": 0, "tie": 0, "skip": 0}
    pairs, raters = [], {}
    if not LOG.exists():
        return pairs, counts, raters
    with LOG.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ch = r.get("choice", "")
            raters[r.get("rater", "?")] = raters.get(r.get("rater", "?"), 0) + (ch in ("L", "R"))
            if ch in ("L", "R") and r.get("winner_seg") and r.get("loser_seg"):
                pairs.append((int(r["winner_seg"]), int(r["loser_seg"])))
                counts["decisive"] += 1
            elif ch == "tie":
                counts["tie"] += 1
            elif ch == "skip":
                counts["skip"] += 1
    return pairs, counts, raters


def diffs(pairs, Z):
    W = Z.loc[[w for w, _ in pairs]].to_numpy()
    L = Z.loc[[l for _, l in pairs]].to_numpy()
    return W - L


def fit_coef(pairs, Z, C=1.0):
    d = diffs(pairs, Z)
    X = np.vstack([d, -d])
    y = np.r_[np.ones(len(d)), np.zeros(len(d))]
    return LogisticRegression(fit_intercept=False, C=C, max_iter=2000).fit(X, y).coef_[0]


def cv(pairs, Z, latlon, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(latlon.to_numpy())
    blk = dict(zip(latlon.index, km))
    correct, held = [], 0
    for f in range(k):
        test = [(w, l) for w, l in pairs if blk.get(w) == f and blk.get(l) == f]
        train = [(w, l) for w, l in pairs if blk.get(w) != f and blk.get(l) != f]
        if not test or len(train) < MIN_PAIRS:
            continue
        held += len(test)
        coef = fit_coef(train, Z)
        correct.extend((diffs(test, Z) @ coef > 0).tolist())
    return (round(float(np.mean(correct)), 3) if correct else None), held


def single_feature_agreement(pairs, raw):
    out = {}
    for c in FEATS:
        wv = np.array([raw.loc[w, c] for w, _ in pairs], dtype=float)
        lv = np.array([raw.loc[l, c] for _, l in pairs], dtype=float)
        m = wv != lv
        out[c] = {"agreement": round(float((wv > lv)[m].mean()), 3), "n_differ": int(m.sum())}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["agreement"]))


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(spearmanr(a[m], b[m]).statistic), 3) if m.sum() > 30 else None


def main():
    Z, meta = load_Z()
    raw = pd.read_parquet(PROC / "la_seg_features.parquet").set_index("seg_id")
    pairs, counts, raters = load_pairs()
    if len(pairs) < MIN_PAIRS:
        print(f"Only {len(pairs)} decisive LA pairs (need >= {MIN_PAIRS}). Rate via "
              "python scripts/blind_compare.py --city la")
        return

    coef = fit_coef(pairs, Z)
    rng = np.random.default_rng(SEED)
    boots = np.array([fit_coef([pairs[i] for i in rng.integers(0, len(pairs), len(pairs))], Z)
                      for _ in range(300)])
    weights = {c: {"coef": round(float(coef[i]), 3),
                   "ci95": [round(float(np.percentile(boots[:, i], 2.5)), 3),
                            round(float(np.percentile(boots[:, i], 97.5)), 3)],
                   "abs_weight": round(float(abs(coef[i]) / (np.abs(coef).sum() + 1e-9)), 3)}
               for i, c in enumerate(FEATS)}

    sfa = single_feature_agreement(pairs, raw)
    sweep = []
    for k in CV_BLOCKS:
        ag, held = cv(pairs, Z, meta[["mx", "my"]], k)
        sweep.append({"blocks": k, "held_out": held, "agreement": ag})

    # learned comfort score across all LA segments, then bias-audit vs income + race
    comfort = (Z[FEATS].to_numpy() @ coef)
    cs = pd.Series(comfort, index=Z.index)
    inc = meta["income"].to_numpy(dtype=float)
    nw = meta["pct_nonwhite"].to_numpy(dtype=float)
    csv_ = cs.to_numpy()
    bias = {
        "learned_comfort_vs_income": sp(csv_, inc),
        "learned_comfort_vs_pct_nonwhite": sp(csv_, nw),
        "per_feature_vs_income": {c: sp(Z[c].to_numpy(), inc) for c in FEATS},
        "per_feature_vs_pct_nonwhite": {c: sp(Z[c].to_numpy(), nw) for c in FEATS},
    }

    bi, bn = bias["learned_comfort_vs_income"], bias["learned_comfort_vs_pct_nonwhite"]
    mxbias = max(abs(bi or 0), abs(bn or 0))
    if mxbias < 0.10:
        lvl, tail = "demographically CLEAN", ("Close to zero, like the Boise features; the diverse-"
            "city judgments did not encode income or race.")
    elif mxbias < 0.20:
        lvl, tail = "MOSTLY clean with a mild, borderline tilt", ("The tilt is mild but real and "
            "comes from valuing business density, which skews slightly higher-income / whiter in "
            "this transect. Not a strong proxy, but worth monitoring with more raters and cities.")
    else:
        lvl, tail = "a demographic PROXY", "Strong enough to flag; the comfort score tracks income/race."
    verdict = (f"The rater's LA 'safer' calls yield a comfort score that is {lvl} "
               f"(vs income {bi}, vs %nonwhite {bn}). " + tail)
    caveats = [
        "Single rater (BJ); daytime photos; not the target demographic.",
        "LA lit_frac variance is compressed (median 1.0), so the lighting weight is unreliable here; "
        "this round is driven by business density and built form.",
        "No vision model on LA, so this is structured-only (no B1/C comparison).",
        f"Spatial held-out is small ({max(s['held_out'] for s in sweep)} of {len(pairs)} pairs); CIs wide.",
        "Crime/demographics are audit targets only, never features; weights are learned.",
    ]
    out = {"city": "Los Angeles (Central/South transect)", "rater_counts": raters,
           "n_decisive": len(pairs), "counts": counts, "weights": weights,
           "robust_single_feature_agreement": sfa, "spatial_cv_sensitivity": sweep,
           "bias_audit": bias, "verdict": verdict, "caveats": caveats}
    (PROC / "la_fit.json").write_text(json.dumps(out, indent=2))

    print(f"LA fit | rater(s) {raters} | decisive {len(pairs)} (ties {counts['tie']})")
    print("\nlearned weights (standardized logistic; +coef => safer-feeling):")
    for c in FEATS:
        w = weights[c]
        sig = "" if (w["ci95"][0] <= 0 <= w["ci95"][1]) else "  *signif*"
        print(f"  {c:18s} {w['coef']:+.2f}  CI[{w['ci95'][0]:+.2f},{w['ci95'][1]:+.2f}]{sig}")
    print("\nrobust single-feature agreement (all decisive pairs):")
    for c, d in sfa.items():
        print(f"  pick higher {c:18s} {d['agreement']:.0%}  (n={d['n_differ']})")
    print("\nspatial-block CV (held-out agreement):")
    for s in sweep:
        print(f"  blocks {s['blocks']}  held {s['held_out']:>3}  agreement {s['agreement']}")
    print(f"\nBIAS AUDIT of the LEARNED comfort score:")
    print(f"  vs income      {bi}")
    print(f"  vs %nonwhite   {bn}")
    print(f"\nVERDICT: {verdict}")
    print("wrote la_fit.json")


if __name__ == "__main__":
    main()

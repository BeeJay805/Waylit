"""Stage 2: on-demand night-walking-comfort engine for any OSM+Overture area.

Loads the Stage 1 transferable model (data/processed/transferable_comfort_weights.json) and
scores per-segment night-comfort for an arbitrary location from 4 OSM+Overture features only
(poi_density, poi_night_density, encl_frontage, encl_height) -- no imagery, no per-city AI
rating. Standardizes WITHIN the queried area (relative comfort) exactly as Stage 1 did, then
applies the pooled weights and emits an out-of-distribution confidence flag (urban form unlike
the dense-US-downtown training set -> low confidence, degrade gracefully).

This is the AI-panel-derived layer, SEPARATE from comfort_weights.json (human routing weights),
which it never reads or writes. Crime/demographics are never used. Not a crime score; it never
labels a place safe/dangerous; it estimates PERCEIVED comfort from street structure.

Modes (one required):
  --features PARQUET [--tag NAME]                 score an already-built feature parquet (fast)
  --bbox minlon,minlat,maxlon,maxlat [--tag NAME] build (OSM+Overture) then score a bbox
  --lat LAT --lon LON --radius_m M [--tag NAME]   build then score a square around a point
Options:
  --metric EPSG:xxxx   metric CRS for the build (default: auto UTM from the area center)
  --map                also render <tag>_comfort.png

Outputs (data/processed/engine/):
  <tag>_comfort.parquet  seg_id, lat, lon, 4 features, comfort_latent, comfort_pct, ood_*, confidence
  <tag>_comfort.json     area summary: confidence, coverage, OOD fraction, weights + decision used

Examples:
  python scripts/comfort_engine.py --features data/processed/segment_features.parquet --tag boise
  python scripts/comfort_engine.py --lat 45.5202 --lon -122.6742 --radius_m 700 --tag portland_dt --map
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ENGINE = PROC / "engine"
WEIGHTS = PROC / "transferable_comfort_weights.json"
sys.path.insert(0, str(ROOT / "scripts"))


def load_model():
    if not WEIGHTS.exists():
        sys.exit(f"missing {WEIGHTS} -- run: python scripts/stage1_transferability.py")
    return json.loads(WEIGHTS.read_text())


def transform(x, col, log1p):
    return np.log1p(np.clip(x, 0, None)) if col in log1p else x.astype(float)


def score(df, model):
    """Within-area standardize (matches Stage 1), apply pooled weights, flag out-of-distribution."""
    feats, log1p = model["features"], set(model["log1p"])
    coef = np.array([model["coef"][f] for f in feats])
    st = model["training_raw_stats_log1p_scale"]

    Z = np.zeros((len(df), len(feats)))        # within-area z-score (relative comfort)
    Zood = np.zeros((len(df), len(feats)))     # vs training distribution (absolute, for OOD)
    for j, c in enumerate(feats):
        t = transform(df[c].to_numpy(dtype=float), c, log1p)
        Z[:, j] = (t - t.mean()) / (t.std() + 1e-9)
        Zood[:, j] = (t - st[c]["mean"]) / st[c]["std"]
    latent = Z @ coef
    dist = np.sqrt((Zood ** 2).mean(axis=1))
    ood = dist > model["ood"]["cutoff_p95"]
    # any-business presence; building frontage is ~1.0 on every US grid so it cannot discriminate
    context = df["poi_density"].to_numpy() > 0

    out = df.copy()
    out["comfort_latent"] = latent.round(4)
    out["comfort_pct"] = (pd.Series(latent).rank(pct=True).to_numpy() * 100).round(1)
    out["ood_distance"] = dist.round(3)
    out["ood_flag"] = ood
    out["has_context"] = context
    out["confidence"] = np.where(ood | ~context, "low", "high")
    return out


def summarize(out, model, tag, info):
    ood_frac = float(out.ood_flag.mean())
    coverage = float(out.has_context.mean())                         # share with any business nearby
    night_cov = float((out.poi_night_density.to_numpy() > 0).mean())  # share with night business
    # The model's dominant signal is night-business (coef +0.46). An area whose night-activity share
    # is far below the training downtowns lacks the core the model was validated on -> low confidence,
    # even when individual streets look ordinary (a quiet suburban street ~ a quiet downtown street).
    train_night = model.get("area_reference", {}).get("night_activity_share", 0.66)
    activity_sufficiency = round(min(1.0, night_cov / train_night), 3)
    conf = round(activity_sufficiency * (1 - ood_frac), 2)
    label = "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low"
    return {
        "tag": tag, "n_segments": int(len(out)),
        "area_confidence": conf, "area_confidence_label": label,
        "activity_sufficiency_vs_training": activity_sufficiency,
        "night_activity_coverage": round(night_cov, 3),
        "training_night_activity_share": train_night,
        "coverage_any_business": round(coverage, 3),
        "out_of_distribution_fraction": round(ood_frac, 3),
        "comfort_latent_range": [round(float(out.comfort_latent.min()), 3),
                                 round(float(out.comfort_latent.max()), 3)],
        "build_info": info,
        "model_features": model["features"], "model_coef": model["coef"],
        "model_decision": model["decision"],
        "note": "comfort_pct is RELATIVE within this area (0=calmest-feeling..100=most-active). "
                "AI perceived-comfort from structure, not human ground truth; not a crime score. "
                "Low area_confidence => the area is unlike the dense-downtown training set; scores degrade.",
    }


def utm_epsg(lon, lat):
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


def bbox_from_point(lat, lon, radius_m):
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return {"min_lon": lon - dlon, "min_lat": lat - dlat,
            "max_lon": lon + dlon, "max_lat": lat + dlat}


def render_map(out, tag, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    hi = out[out.confidence == "high"]
    lo = out[out.confidence == "low"]
    ax.scatter(lo.lon, lo.lat, c="#cccccc", s=10, label="low confidence (OOD / no context)")
    sc = ax.scatter(hi.lon, hi.lat, c=hi.comfort_pct, cmap="viridis", s=14, vmin=0, vmax=100)
    fig.colorbar(sc, ax=ax, label="night-comfort percentile (relative within area)")
    ax.set_title(f"Stage 2 engine: night-walking-comfort -- {tag}\n"
                 "relative within-area perceived comfort from 4 OSM/Overture features "
                 "(NOT a crime score)", fontsize=10)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="On-demand night-walking-comfort engine")
    ap.add_argument("--features"); ap.add_argument("--bbox")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--radius_m", type=float, default=700)
    ap.add_argument("--metric"); ap.add_argument("--tag")
    ap.add_argument("--map", action="store_true")
    args = ap.parse_args()
    ENGINE.mkdir(parents=True, exist_ok=True)
    model = load_model()

    if args.features:
        tag = args.tag or pathlib.Path(args.features).stem
        df = pd.read_parquet(args.features)
        info = {"source": "prebuilt features", "path": args.features, "walk_segments": int(len(df))}
    else:
        if args.bbox:
            mn_lon, mn_lat, mx_lon, mx_lat = (float(x) for x in args.bbox.split(","))
            bbox = {"min_lon": mn_lon, "min_lat": mn_lat, "max_lon": mx_lon, "max_lat": mx_lat}
        elif args.lat is not None and args.lon is not None:
            bbox = bbox_from_point(args.lat, args.lon, args.radius_m)
        else:
            ap.error("provide one of --features, --bbox, or --lat/--lon")
        cx = (bbox["min_lon"] + bbox["max_lon"]) / 2
        cy = (bbox["min_lat"] + bbox["max_lat"]) / 2
        metric = args.metric or utm_epsg(cx, cy)
        tag = args.tag or f"area_{cy:.3f}_{cx:.3f}"
        from build_city_features import build_features
        print(f"building features for {tag} (metric {metric}) ...")
        df, info = build_features(bbox, metric, tag)
        info["bbox"] = bbox; info["metric"] = metric

    missing = [c for c in model["features"] if c not in df.columns]
    if missing:
        sys.exit(f"feature parquet missing model columns: {missing}")

    out = score(df, model)
    summary = summarize(out, model, tag, info)
    pq = ENGINE / f"{tag}_comfort.parquet"
    js = ENGINE / f"{tag}_comfort.json"
    keep = ["seg_id", "lat", "lon", *model["features"], "comfort_latent", "comfort_pct",
            "ood_distance", "ood_flag", "has_context", "confidence"]
    out[[c for c in keep if c in out.columns]].to_parquet(pq)
    js.write_text(json.dumps(summary, indent=2))
    if args.map:
        render_map(out, tag, ENGINE / f"{tag}_comfort.png")

    print(f"\n{tag}: {summary['n_segments']} segments | area confidence {summary['area_confidence']} "
          f"({summary['area_confidence_label']}) | night-activity coverage {summary['night_activity_coverage']} "
          f"(train {summary['training_night_activity_share']}) | OOD frac {summary['out_of_distribution_fraction']}")
    top = out[out.confidence == "high"].nlargest(5, "comfort_pct")
    print("highest-comfort (within-area) segments:")
    for _, r in top.iterrows():
        print(f"  seg {int(r.seg_id):>5}  pct {r.comfort_pct:>5.1f}  night_biz {int(r.poi_night_density):>3}  "
              f"biz {int(r.poi_density):>3}  frontage {r.encl_frontage:.2f}  ({r.lat:.5f},{r.lon:.5f})")
    print(f"wrote {pq.name}, {js.name}" + (f", {tag}_comfort.png" if args.map else ""))


if __name__ == "__main__":
    main()

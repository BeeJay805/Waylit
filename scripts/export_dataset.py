"""Export the open Waylit dataset (v1) from the unified feature table + walk-safety layer.

v1 adds, on top of the v0 features: the day-walking-safety layer (objective, traffic-based) and
two transparent scores:
  comfort_night  calibrated from blind pairwise night judgments (comfort_weights.json A_structured)
  day_safety     objective, reasoned from pedestrian-safety evidence, crash-validated (ordering)

Both scores are bootstrap/transparent, not final (single night rater; reasoned day constants). See
dataset/DATA_DICTIONARY.md. Crime/demographics are never inputs.

  dataset/waylit_boise_downtown_v1.csv         features + both scores + WGS84 geometry (WKT), tracked
  data/processed/release/..._v1.geojson         same with geometry (regenerable, git-ignored)
"""
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
NIGHT_LOG1P = {"encl_height", "poi_density", "poi_night_density"}
BASE = ["seg_id", "u", "v", "key", "street", "highway", "length_m", "lat", "lon",
        "light", "encl_frontage", "encl_height", "poi_density", "poi_night_density",
        "transit_night_400m", "vis_sidewalk", "vis_vegetation", "vis_sky",
        "vis_building", "vis_pole", "has_visual"]
DAY = ["car_free", "traffic_class", "maxspeed_mph", "lanes", "traffic_exposure",
       "sidewalk_present", "crossing_near", "day_safety"]


def night_comfort(feat):
    """Calibrated night comfort 0..1: learned weights dotted with standardized features."""
    w = json.loads((PROC / "comfort_weights.json").read_text())["weights"]["A_structured"]
    raw = np.zeros(len(feat))
    for f, meta in w.items():
        x = feat[f].astype(float).to_numpy()
        if f in NIGHT_LOG1P:
            x = np.log1p(np.clip(x, 0, None))
        raw += meta["coef"] * (x - x.mean()) / (x.std() + 1e-9)
    p5, p95 = np.percentile(raw, [5, 95])
    return np.clip((raw - p5) / (p95 - p5 + 1e-9), 0, 1)


def main():
    feat = gpd.read_file(PROC / "segment_features.gpkg").to_crs("EPSG:4326")
    ws = pd.read_parquet(PROC / "walk_safety_features.parquet")[["seg_id"] + DAY]
    df = pd.DataFrame(feat[[c for c in BASE if c in feat.columns]]).merge(ws, on="seg_id", how="left")
    df["comfort_night"] = night_comfort(feat)

    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4)
    df["geometry_wkt"] = feat.geometry.to_wkt()

    out_dir = ROOT / "dataset"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "waylit_boise_downtown_v1.csv"
    df.to_csv(csv_path, index=False)
    # drop the superseded v0 so the release is unambiguous
    old = out_dir / "waylit_boise_downtown_v0.csv"
    if old.exists():
        old.unlink()

    rel = PROC / "release"
    rel.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(df.drop(columns="geometry_wkt"), geometry=feat.geometry.values, crs="EPSG:4326")
    gdf.to_file(rel / "waylit_boise_downtown_v1.geojson", driver="GeoJSON")

    print(f"rows: {len(df)} | columns: {len(df.columns)}")
    print(f"CSV: {csv_path.name} ({csv_path.stat().st_size // 1024} KB)")
    print(f"day_safety median {df.day_safety.median():.2f} | comfort_night median {df.comfort_night.median():.2f} "
          f"| with visual {int(df['has_visual'].sum())}")


if __name__ == "__main__":
    main()

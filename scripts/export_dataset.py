"""Export the open Waylit dataset (v0) from the unified feature table.

Writes a self-contained, documented release:
  dataset/waylit_boise_downtown_v0.csv        features + WGS84 geometry (WKT), git-tracked
  data/processed/release/..._v0.geojson        same, with geometry (regenerable, git-ignored)

This is a DRAFT: weights are uncalibrated and the scores are NOT validated against measured
night conditions yet (that needs the pilot). See dataset/DATA_DICTIONARY.md.
"""
import pathlib

import geopandas as gpd
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLS = ["seg_id", "u", "v", "key", "street", "highway", "length_m", "lat", "lon",
        "light", "encl_frontage", "encl_height", "poi_density", "poi_night_density",
        "transit_night_400m", "vis_sidewalk", "vis_vegetation", "vis_sky",
        "vis_building", "vis_pole", "has_visual"]


def main():
    feat = gpd.read_file(ROOT / "data/processed/segment_features.gpkg").to_crs("EPSG:4326")
    cols = [c for c in COLS if c in feat.columns]
    df = pd.DataFrame(feat[cols]).copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4)
    df["geometry_wkt"] = feat.geometry.to_wkt()

    out_dir = ROOT / "dataset"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "waylit_boise_downtown_v0.csv"
    df.to_csv(csv_path, index=False)

    rel = ROOT / "data" / "processed" / "release"
    rel.mkdir(parents=True, exist_ok=True)
    feat[cols + ["geometry"]].to_file(rel / "waylit_boise_downtown_v0.geojson", driver="GeoJSON")

    print(f"rows: {len(df)} | columns: {len(df.columns)}")
    print(f"CSV: {csv_path} ({csv_path.stat().st_size // 1024} KB)")
    print("with visual features:", int(df["has_visual"].sum()) if "has_visual" in df else "n/a")


if __name__ == "__main__":
    main()

"""Join enumerated Mapillary images to walk-graph segments.

Uses data/raw/mapillary/images.jsonl (image coordinates) + data/processed/system_a_edges.gpkg
(street segments). Reports segment-level imagery coverage and the leaf-on/leaf-off mix
(which matters for the canopy-occlusion feature). No downloads.

Run:  python scripts/segment_imagery_join.py
"""
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mapillary" / "images.jsonl"
EDGES = ROOT / "data" / "processed" / "system_a_edges.gpkg"
METRIC = "EPSG:32611"
MAX_DIST = 20.0  # metres: an image counts for a segment if within this distance


def main():
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    imgs = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(METRIC)

    edges = gpd.read_file(EDGES)
    edges["__wkb"] = edges.geometry.to_wkb()
    edges = edges.drop_duplicates("__wkb").reset_index(drop=True)
    edges["seg_id"] = np.arange(len(edges))

    joined = gpd.sjoin_nearest(imgs, edges[["seg_id", "geometry"]],
                               max_distance=MAX_DIST, distance_col="d")
    counts = joined.groupby("seg_id").size()
    edges["n_img"] = edges["seg_id"].map(counts).fillna(0).astype(int)

    nseg = len(edges)
    covered = int((edges["n_img"] > 0).sum())
    matched = int(joined["seg_id"].notna().sum())
    months = pd.to_datetime(df["captured_at"], unit="ms", errors="coerce").dt.month
    leaf_on = float(months.between(4, 10).mean())

    summary = {
        "unique_segments": nseg,
        "segments_with_imagery": covered,
        "segment_coverage_pct": round(100 * covered / nseg, 1),
        "images_matched_to_a_segment": matched,
        "images_total": len(df),
        "median_images_per_covered_segment": int(edges.loc[edges.n_img > 0, "n_img"].median()),
        "segments_with_3plus_images": int((edges["n_img"] >= 3).sum()),
        "leaf_on_fraction_apr_oct": round(leaf_on, 2),
    }
    out = ROOT / "data" / "processed" / "segment_imagery.gpkg"
    try:
        edges[["seg_id", "light", "n_img", "geometry"]].to_file(out, driver="GPKG")
    except Exception as e:
        print("(gpkg write skipped:", e, ")")
    (ROOT / "data" / "processed" / "segment_imagery_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Map each segment to the daytime photo(s) that produced its visual features.

The unified table's vis_* columns came from spatial-joining CV-processed Mapillary
images to segments within 20 m (see build_features.visual). That join is not persisted,
so the blind pairwise tool has no way to show "the photo this segment was scored from".
This script replays the identical join but keeps the image ids and their distance, so the
shown photo is exactly the imagery the model sees. No downloads, no scores.

Output: data/processed/segment_image_link.parquet
  seg_id, img_best (nearest on-disk id), img_ids (list, nearest first), n_img, dist_best_m
"""
import json
import pathlib

import geopandas as gpd
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
IMGDIR = ROOT / "data" / "raw" / "mapillary" / "img"
METRIC = "EPSG:32611"
MAX_DIST = 20.0  # metres, identical to the feature-table join


def main():
    segs = gpd.read_file(OUT / "segment_features.gpkg")[["seg_id", "geometry"]].to_crs(METRIC)

    # only images that were actually CV-processed (guaranteed on disk + fed vis_*)
    feats = [json.loads(x) for x in (OUT / "visual_features.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    ids = pd.DataFrame(feats)[["id"]].drop_duplicates()
    coords = {r["id"]: (r["lon"], r["lat"]) for r in (json.loads(x) for x in
              (ROOT / "data/raw/mapillary/images.jsonl").read_text(encoding="utf-8").splitlines())}
    ids["lon"] = ids["id"].map(lambda i: coords.get(i, (None, None))[0])
    ids["lat"] = ids["id"].map(lambda i: coords.get(i, (None, None))[1])
    ids = ids.dropna(subset=["lon", "lat"])
    ids["on_disk"] = ids["id"].map(lambda i: (IMGDIR / f"{i}.jpg").exists())
    ids = ids[ids["on_disk"]].copy()

    fg = gpd.GeoDataFrame(ids, geometry=gpd.points_from_xy(ids.lon, ids.lat),
                          crs="EPSG:4326").to_crs(METRIC)
    j = gpd.sjoin_nearest(fg, segs, max_distance=MAX_DIST, distance_col="d")
    j = j.sort_values(["seg_id", "d"])

    rows = []
    for seg_id, grp in j.groupby("seg_id"):
        rows.append({
            "seg_id": int(seg_id),
            "img_best": str(grp.iloc[0]["id"]),
            "img_ids": [str(x) for x in grp["id"].tolist()],
            "n_img": int(len(grp)),
            "dist_best_m": round(float(grp.iloc[0]["d"]), 1),
        })
    link = pd.DataFrame(rows).sort_values("seg_id").reset_index(drop=True)
    link.to_parquet(OUT / "segment_image_link.parquet")

    df = pd.read_parquet(OUT / "segment_features.parquet")
    hv = set(df.loc[df.has_visual, "seg_id"])
    linked = set(link["seg_id"])
    print(f"linked segments: {len(link)} | has_visual: {len(hv)} | "
          f"match: {len(linked & hv)} | has_visual missing a photo: {len(hv - linked)}")
    print(f"median photos/segment: {int(link['n_img'].median())} | "
          f"median nearest dist: {link['dist_best_m'].median():.1f} m")
    print("saved segment_image_link.parquet")


if __name__ == "__main__":
    main()

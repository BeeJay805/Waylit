"""Add a transit-at-night feature to the unified feature table.

Downloads the Valley Regional Transit GTFS feed, finds stops with evening/late service
(any departure at hour >= 20, including after-midnight 24+ codes), and for each segment
records how many night-active stops are within a 400 m walk. Boise night service is
limited, so many segments will legitimately read 0 (itself a real signal).

Updates data/processed/segment_features.{gpkg,parquet} in place (adds transit_night_400m).
"""
import pathlib
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRIC = "EPSG:32611"
GTFS_URL = "https://s3.amazonaws.com/etatransit.gtfs/valleyregionaltransit.etaspot.net/gtfs.zip"
GTFS_DIR = ROOT / "data" / "raw" / "gtfs"
NIGHT_HOUR = 20
WALK_R = 400.0


def night_stops():
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    zp = GTFS_DIR / "vrt_gtfs.zip"
    if not zp.exists() or zp.stat().st_size == 0:
        urllib.request.urlretrieve(GTFS_URL, str(zp))
    with zipfile.ZipFile(zp) as z:
        stops = pd.read_csv(z.open("stops.txt"))
        st = pd.read_csv(z.open("stop_times.txt"), usecols=["stop_id", "departure_time"],
                         dtype={"stop_id": str})
    st["hr"] = pd.to_numeric(st["departure_time"].astype(str).str.split(":").str[0], errors="coerce")
    latest = int(st["hr"].max())
    stop_max = st.groupby("stop_id")["hr"].max()
    night_ids = set(stop_max[stop_max >= NIGHT_HOUR].index.astype(str))
    stops["stop_id"] = stops["stop_id"].astype(str)
    ns = stops[stops["stop_id"].isin(night_ids)].dropna(subset=["stop_lat", "stop_lon"])
    g = gpd.GeoDataFrame(ns, geometry=gpd.points_from_xy(ns.stop_lon, ns.stop_lat),
                         crs="EPSG:4326").to_crs(METRIC)
    return g, len(stops), latest


def main():
    ns, n_stops, latest = night_stops()
    print(f"GTFS: {n_stops} stops total | latest scheduled departure hour: {latest}")
    print(f"night-active stops (departure hour >= {NIGHT_HOUR}): {len(ns)}")

    feat = gpd.read_file(ROOT / "data/processed/segment_features.gpkg")
    mids = feat.geometry.interpolate(0.5, normalized=True)
    mx, my = mids.x.to_numpy(), mids.y.to_numpy()
    if len(ns):
        tree = cKDTree(np.c_[ns.geometry.x.to_numpy(), ns.geometry.y.to_numpy()])
        feat["transit_night_400m"] = [len(tree.query_ball_point((x, y), WALK_R)) for x, y in zip(mx, my)]
    else:
        feat["transit_night_400m"] = 0

    feat.to_file(ROOT / "data/processed/segment_features.gpkg", driver="GPKG")
    pd.DataFrame(feat.drop(columns="geometry")).to_parquet(ROOT / "data/processed/segment_features.parquet")
    share = (feat["transit_night_400m"] > 0).mean()
    print(f"segments with a night-active stop within {WALK_R:.0f} m: {share:.0%}")
    print("added transit_night_400m to segment_features.{gpkg,parquet}")


if __name__ == "__main__":
    main()

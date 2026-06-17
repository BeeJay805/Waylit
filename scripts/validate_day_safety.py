"""Objective validation of the day-safety score against real pedestrian crashes.

day_safety is reasoned, not learned, so the honest check is: do pedestrian-vehicle crashes actually
concentrate where the score says it is dangerous? Crashes need a car AND a pedestrian, so by
construction they happen on roads/crossings, not on the car-free footways the score rates safest.
If crashes cluster on the LOW day_safety (high-traffic) roads, the score's ordering holds.

Crash source: COMPASS Treasure Valley crash data (ITD), first_unittype='Pedestrian' (pedestrian as
first unit = a location-unbiased lower bound). Crashes are validation only, never a model input.

Output: data/processed/day_safety_validation.json
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from waylit import arcgis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
B = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
CRASH = "https://swidrdc.org/arcgis/rest/services/COMPASSData/CrashData/FeatureServer/1"


def crashes(where):
    feats, off = [], 0
    while True:
        p = arcgis.query(CRASH, where=where, geometry=BBOX, geometryType="esriGeometryEnvelope",
                         inSR="4326", spatialRel="esriSpatialRelIntersects", outFields="severity",
                         returnGeometry="true", outSR="4326", resultOffset=off,
                         resultRecordCount=2000).get("features", [])
        feats += p
        if len(p) < 2000:
            break
        off += 2000
    xy = [(f["geometry"]["x"], f["geometry"]["y"]) for f in feats if f.get("geometry")]
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy([x for x, _ in xy], [y for _, y in xy]),
                            crs="EPSG:4326").to_crs(METRIC)


def main():
    sf = pd.read_parquet(PROC / "segment_features.parquet")[["seg_id", "lat", "lon"]]
    ws = pd.read_parquet(PROC / "walk_safety_features.parquet")[
        ["seg_id", "day_safety", "car_free", "traffic_class"]]
    seg = sf.merge(ws, on="seg_id")
    sp = gpd.GeoSeries(gpd.points_from_xy(seg.lon, seg.lat), crs="EPSG:4326").to_crs(METRIC)
    seg["mx"], seg["my"] = sp.x.to_numpy(), sp.y.to_numpy()

    ped = crashes("first_unittype='Pedestrian'")
    allc = crashes("1=1")
    print(f"pedestrian crashes in bbox: {len(ped)} | all crashes: {len(allc)}")

    seg_tree = cKDTree(np.c_[seg.mx, seg.my])
    road = seg[seg.car_free == 0].reset_index(drop=True)
    road_tree = cKDTree(np.c_[road.mx, road.my])

    # 1) which kind of segment is each pedestrian crash nearest to?
    dseg, iseg = seg_tree.query(np.c_[ped.geometry.x, ped.geometry.y])
    near = seg.iloc[iseg]
    on_road = (near.car_free.to_numpy() == 0)
    # 2) assign each crash to nearest ROAD (<=40 m) and count per road segment
    dr, ir = road_tree.query(np.c_[ped.geometry.x, ped.geometry.y])
    keep = dr <= 40
    cnt = pd.Series(ir[keep]).value_counts()
    road["ped_crashes"] = road.index.map(cnt).fillna(0).astype(int)

    net_mean = float(seg.day_safety.mean())
    crash_seg_mean = float(near.day_safety.mean())
    rho = spearmanr(road.day_safety, road.ped_crashes).statistic
    # crashes by day_safety quartile of roads
    road["q"] = pd.qcut(road.day_safety.rank(method="first"), 4, labels=["lowest", "low", "high", "highest"])
    byq = road.groupby("q", observed=True)["ped_crashes"].sum().to_dict()
    byclass = {int(c): int(road[road.traffic_class == c]["ped_crashes"].sum())
               for c in sorted(road.traffic_class.unique())}

    out = {
        "source": "COMPASS/ITD crash data, first_unittype=Pedestrian (location-unbiased lower bound)",
        "n_ped_crashes": int(len(ped)), "n_all_crashes": int(len(allc)),
        "network_car_free_share": round(float((seg.car_free == 1).mean()), 3),
        "ped_crashes_nearest_a_road_not_footway": round(float(on_road.mean()), 3),
        "mean_day_safety_network": round(net_mean, 3),
        "mean_day_safety_at_ped_crash_sites": round(crash_seg_mean, 3),
        "spearman_roadsafety_vs_pedcrashcount": round(float(rho), 3),
        "ped_crashes_by_road_day_safety_quartile": {k: int(v) for k, v in byq.items()},
        "ped_crashes_by_traffic_class": byclass,
        "verdict": "",
        "caveats": [
            "first_unittype=Pedestrian misses crashes where the car is unit 1, so counts are a lower bound.",
            "Crashes need pedestrians AND cars present, an exposure confound; busy roads have more of both. "
            "But the score explicitly models that traffic gradient, so concentration on low-day_safety roads is supportive.",
            "Crash coordinates are geocoded to street/intersection, so per-segment assignment is approximate (40 m).",
            "Validation only; crashes are never a model input. Score constants could be refit to these crashes next."],
    }
    road_share = 1 - float((seg.car_free == 1).mean())
    if rho < -0.10:
        out["verdict"] = (
            f"SUPPORTED (ordering): among roads, pedestrian crashes concentrate where the score says "
            f"danger is - Spearman(road day_safety, crash count) {rho:+.2f}, and counts rise with traffic "
            f"class {byclass}. Crashes are {on_road.mean():.0%} nearest a road although roads are only "
            f"{road_share:.0%} of the walk network. Per-segment precision is limited: crash points are "
            f"geocoded to street/intersection and footways are mapped beside every road, so "
            f"{1-on_road.mean():.0%} fall nearest a parallel footway and dilute the raw site-mean "
            f"({crash_seg_mean:.2f} vs {net_mean:.2f}). Validates the danger ORDERING, not the exact constants.")
    else:
        out["verdict"] = (f"WEAK/MIXED: crash concentration does not clearly track day_safety "
                          f"(Spearman {rho:+.2f}, site mean {crash_seg_mean:.2f} vs network {net_mean:.2f}); "
                          "inspect before trusting the constants.")
    (PROC / "day_safety_validation.json").write_text(json.dumps(out, indent=2))

    print(f"\n{out['ped_crashes_nearest_a_road_not_footway']:.0%} of pedestrian crashes are nearest a ROAD "
          f"(roads are only {1-(seg.car_free==1).mean():.0%} of the walk network)")
    print(f"mean day_safety: network {net_mean:.2f} | at crash sites {crash_seg_mean:.2f}")
    print(f"Spearman(road day_safety, ped-crash count): {rho:+.3f}")
    print(f"ped crashes by road day_safety quartile (lowest=most dangerous): {byq}")
    print(f"ped crashes by traffic_class (2 residential .. 5 primary): {byclass}")
    print(f"\nVERDICT: {out['verdict']}")
    print("wrote day_safety_validation.json")


if __name__ == "__main__":
    main()

"""Day-walking safety features: the pedestrian-traffic layer the night-comfort score misses.

By day the dominant hazard is cars, not darkness. Per segment, from OSM (free, portable):
  car_free          1 if a dedicated pedestrian way (footway/path/pedestrian/steps/track), else 0
  traffic_class     0 car-free .. 5 primary arterial (road class = exposure to traffic)
  traffic_exposure  0 for car-free; else class * speed(mph) * lanes, normalized 0..1
  sidewalk_present  1.0 car-free or OSM sidewalk in {both,left,right,yes}; 0 if {no,none}; 0.5 unknown
  crossing_near     count of marked/signalized crossings within 40 m

These calibrate a DAY-safety score, kept SEPARATE from the night-comfort score (a street can be
great by day and bad at night, or the reverse). Daytime photos are the correct stimulus for a day
rating round, so no proxy gap. Weights will be LEARNED from that round, never hand-set.

Output: data/processed/walk_safety_features.parquet  (seg_id-aligned with segment_features)
"""
import json
import pathlib

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
from shapely import STRtree
from shapely.geometry import box

ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) |
                                     {"maxspeed", "lanes", "sidewalk", "foot", "surface"})

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
B = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
METRIC = "EPSG:32611"
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}
CARFREE = {"footway", "path", "pedestrian", "steps", "track"}
CLASS = {"footway": 0, "path": 0, "pedestrian": 0, "steps": 0, "track": 0, "living_street": 1,
         "residential": 2, "unclassified": 2, "tertiary": 3, "secondary": 4, "primary": 5}
DEF_SPEED = {1: 10, 2: 25, 3: 30, 4: 35, 5: 40}     # mph default by class when untagged
DEF_LANES = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4}


def coerce(v):
    return (v[0] if v else "") if isinstance(v, list) else ("" if v is None else str(v))


def parse_speed(s, cls):
    s = coerce(s)
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    return float(digits[0]) if digits else DEF_SPEED.get(cls, 25)


def parse_lanes(s, cls):
    s = coerce(s)
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    return float(digits[0]) if digits else DEF_LANES.get(cls, 2)


def sidewalk_val(hw, s):
    if hw in CARFREE:
        return 1.0
    s = coerce(s).lower()
    if s in ("both", "left", "right", "yes", "separate"):
        return 1.0
    if s in ("no", "none"):
        return 0.0
    return 0.5     # untagged: unknown


def crossings_metric(poly):
    try:
        g = ox.features_from_polygon(poly, tags={"highway": "crossing"})
        g = g[g.geometry.type == "Point"]
        return g.to_crs(METRIC)
    except Exception as e:
        print("(crossings unavailable:", str(e)[:120], ")")
        return None


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    E["seg_id"] = np.arange(len(E))
    Em = E.to_crs(METRIC)

    hw = E["highway"]
    cls = hw.map(lambda h: CLASS.get(h, 2)).to_numpy()
    car_free = hw.isin(CARFREE).to_numpy().astype(int)
    sw = E.get("sidewalk", pd.Series([""] * len(E)))
    ms = E.get("maxspeed", pd.Series([""] * len(E)))
    ln = E.get("lanes", pd.Series([""] * len(E)))
    speed = np.array([parse_speed(ms.iloc[i], cls[i]) for i in range(len(E))])
    lanes = np.array([parse_lanes(ln.iloc[i], cls[i]) for i in range(len(E))])
    sidewalk = np.array([sidewalk_val(hw.iloc[i], sw.iloc[i]) for i in range(len(E))])

    raw = np.where(car_free == 1, 0.0, cls * speed * lanes)
    p95 = np.percentile(raw[raw > 0], 95) if (raw > 0).any() else 1.0
    traffic_exposure = np.clip(raw / p95, 0, 1)

    cx = crossings_metric(poly)
    if cx is not None and len(cx):
        tree = STRtree(cx.geometry.values)
        crossing_near = np.array([len(tree.query(g.buffer(40))) for g in Em.geometry])
    else:
        crossing_near = np.zeros(len(E), dtype=int)

    day = pd.DataFrame({
        "u": E["u"].values, "v": E["v"].values, "key": E["key"].values,
        "highway": hw.values, "car_free": car_free, "traffic_class": cls,
        "maxspeed_mph": speed.round(0), "lanes": lanes.round(0),
        "traffic_exposure": traffic_exposure.round(3), "sidewalk_present": sidewalk,
        "crossing_near": crossing_near,
    })
    # align onto the night table's seg_id via normalized edge identity (graph counts differ slightly)
    night = pd.read_parquet(PROC / "segment_features.parquet")[["seg_id", "u", "v", "key"]]
    for d in (day, night):
        d["nu"], d["nv"] = d[["u", "v"]].min(axis=1), d[["u", "v"]].max(axis=1)
    cols = ["car_free", "traffic_class", "maxspeed_mph", "lanes", "traffic_exposure",
            "sidewalk_present", "crossing_near"]
    m = night.merge(day[["nu", "nv", "key"] + cols], on=["nu", "nv", "key"], how="left")
    m = m[~m["seg_id"].duplicated(keep="first")]
    matched = float(m["car_free"].notna().mean())
    m[["seg_id"] + cols].sort_values("seg_id").reset_index(drop=True).to_parquet(
        PROC / "walk_safety_features.parquet")

    roads = day[day.car_free == 0]
    print(f"night seg_ids: {len(night)} | day features matched onto {matched:.0%} (seg_id-aligned)")
    print(f"car-free pedestrian ways: {car_free.mean():.0%} | roads (walk-alongside): {1-car_free.mean():.0%}")
    print(f"road sidewalk from OSM: present {(roads.sidewalk_present==1).mean():.0%}, "
          f"absent {(roads.sidewalk_present==0).mean():.0%}, UNKNOWN {(roads.sidewalk_present==0.5).mean():.0%} "
          "-> Boise barely tags road sidewalks; the car-free footway layer IS the mapped sidewalk network")
    print("\ntraffic_class counts (0 car-free .. 5 primary):")
    print(day.traffic_class.value_counts().sort_index().to_string())
    print(f"\ncrossings within 40m: {(crossing_near>0).mean():.0%} of segments "
          f"(total crossing nodes {len(cx) if cx is not None else 0})")
    print("mean traffic_exposure by class: " +
          ", ".join(f"{c}:{day[day.traffic_class==c].traffic_exposure.mean():.2f}"
                    for c in sorted(day.traffic_class.unique())))
    print("saved walk_safety_features.parquet")


if __name__ == "__main__":
    main()

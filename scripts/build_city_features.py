"""Config-driven, portable feature builder. Proves the pipeline travels to any metro.

Computes the universal (nationally-available) features from OSM + Overture for any city
config in config/cities/<name>.yaml: walk segments, enclosure (frontage + height), and
business density (total + night-active). Lighting is added only where a city publishes a
streetlight inventory (Tier 1); elsewhere it falls to imagery (Tier 3, not run here).

Run:  python scripts/build_city_features.py slc
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from waylit import lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}
NIGHT_KW = ("bar", "pub", "night_club", "brewery", "restaurant", "food", "coffee", "cafe",
            "grocery", "convenience", "pharmacy", "drugstore", "hotel", "motel", "theater",
            "cinema", "liquor", "gas_station", "fitness", "gym")


def coerce(v):
    return (v[0] if v else "") if isinstance(v, list) else ("" if v is None else str(v))


def overture(city, kind, bbox_str, metric):
    p = ROOT / "data" / "raw" / "overture" / f"{city}_{kind}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        cmd = ["download", f"--bbox={bbox_str}", "-f", "geoparquet", "--type", kind, "-o", str(p)]
        try:
            subprocess.run(["overturemaps", *cmd], check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "overturemaps", *cmd], check=True)
    return gpd.read_parquet(p).to_crs(metric)


def enclosure(Em, bld):
    tree = STRtree(bld.geometry.values)
    h = bld["height"].to_numpy(dtype=float) if "height" in bld else np.full(len(bld), np.nan)
    front, mh = [], []
    for geom in Em.geometry:
        pts = lighting._sample_xy(geom, step=18.0)
        hit, hs = 0, []
        for px, py in pts:
            idx = tree.query_nearest(Point(px, py), max_distance=20.0)
            if len(idx):
                hit += 1
                if not np.isnan(h[idx[0]]):
                    hs.append(h[idx[0]])
        front.append(hit / len(pts) if pts else 0.0)
        mh.append(float(np.mean(hs)) if hs else 0.0)
    return np.array(front), np.array(mh)


def build_features(bbox, metric, tag):
    """Build the 4 universal features for an arbitrary bbox. tag keys the Overture cache.
    Returns (df with seg_id + 4 features + lon/lat, info dict). Identical recipe for the
    training cities (via main) and the on-demand engine (comfort_engine.py)."""
    B = bbox
    bbox_str = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'

    G = ox.convert.to_undirected(ox.graph_from_polygon(
        box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"]), network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    Em = E.to_crs(metric)

    bld = overture(tag, "building", bbox_str, metric)
    pl = overture(tag, "place", bbox_str, metric)

    front, mh = enclosure(Em, bld)
    def primary(c):
        if isinstance(c, dict):
            return (c.get("primary") or "")
        try:
            return json.loads(c).get("primary") or ""
        except Exception:
            return ""
    pl["cat"] = pl["categories"].map(primary).str.lower()
    pl["night"] = pl["cat"].map(lambda s: any(k in s for k in NIGHT_KW))
    ptree = STRtree(pl.geometry.values)
    nightarr = pl["night"].to_numpy()
    dens, ndens = [], []
    for geom in Em.geometry:
        idx = ptree.query(geom.buffer(50))
        dens.append(len(idx))
        ndens.append(int(nightarr[idx].sum()) if len(idx) else 0)

    cent = Em.geometry.centroid.to_crs("EPSG:4326")   # segment coords (aligned to seg_id) for imagery
    df = pd.DataFrame({"seg_id": np.arange(len(Em)), "encl_frontage": front.round(3),
                       "encl_height": np.round(mh, 1), "poi_density": dens, "poi_night_density": ndens,
                       "lon": cent.x.values.round(6), "lat": cent.y.values.round(6)})
    info = {"walk_segments": len(Em), "buildings": len(bld),
            "buildings_with_height": float(bld["height"].notna().mean()), "places": len(pl)}
    return df, info


def main(city):
    cfg = yaml.safe_load((ROOT / "config" / "cities" / f"{city}.yaml").read_text())
    METRIC = cfg["crs_metric"]
    print(f"=== {cfg['name']} === metric={METRIC}")
    df, info = build_features(cfg["bbox"], METRIC, city)
    print(f"walk segments: {info['walk_segments']}")
    print(f"Overture: {info['buildings']} buildings ({info['buildings_with_height']:.0%} with height), "
          f"{info['places']} places")
    out = ROOT / "data" / "processed" / f"{city}_features.parquet"
    df.to_parquet(out)
    print(f"median enclosure frontage {df.encl_frontage.median():.2f} | median POI density "
          f"{int(df.poi_density.median())} | median night POIs {int(df.poi_night_density.median())}")
    print(f"lighting: {'Tier-1 inventory available' if cfg.get('streetlight_service') else 'no local inventory (Tier-3 imagery path, not run here)'}")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "slc")

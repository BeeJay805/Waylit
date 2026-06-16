"""Unified per-segment feature table (the backbone).

Builds canonical walk segments once (stable seg_id + OSM u/v/key), then computes every
available feature with provenance, and writes one table the routing, dataset, and model
all read from:
  data/processed/segment_features.gpkg      geometry + features
  data/processed/segment_features.parquet    features only (for ML/joins)

Features now: lighting (Tier-1 inventory), enclosure (Overture buildings), visual
(cached CV), business density + night-active density (Overture places).
Transit-at-night (GTFS) is the next column to add.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402
import yaml  # noqa: E402

from waylit import arcgis, lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
ORG = CFG["arcgis_org_base"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
ENV = "esriGeometryEnvelope"
OUT = ROOT / "data" / "processed"
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}
NIGHT_KW = ("bar", "pub", "night_club", "brewery", "restaurant", "food", "coffee", "cafe",
            "grocery", "convenience", "pharmacy", "drugstore", "hotel", "motel", "theater",
            "cinema", "liquor", "gas_station", "fitness", "gym")
POI_R = 50.0


def coerce(v):
    return (v[0] if v else "") if isinstance(v, list) else ("" if v is None else str(v))


def lamps():
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    feats, off = [], 0
    while True:
        p = arcgis.query(sl, where="Retired_Date IS NULL", outFields="Wattage,Height",
                        geometry=BBOX, geometryType=ENV, inSR="4326",
                        spatialRel="esriSpatialRelIntersects", outSR="4326",
                        returnGeometry="true", resultOffset=off, resultRecordCount=2000).get("features", [])
        feats += p
        if len(p) < 2000:
            break
        off += 2000
    r = [{"Wattage": f["attributes"]["Wattage"], "Height": f["attributes"]["Height"],
          "x": f["geometry"]["x"], "y": f["geometry"]["y"]} for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(pd.DataFrame(r), geometry=gpd.points_from_xy(
        [d["x"] for d in r], [d["y"] for d in r]), crs="EPSG:4326").to_crs(METRIC)
    g["Height"] = g["Height"].astype(float) * 0.3048
    return g


def enclosure(Em, bld):
    tree = STRtree(bld.geometry.values)
    h = bld["height"].to_numpy(dtype=float)
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


def visual(Em):
    feats = [json.loads(x) for x in (OUT / "visual_features.jsonl").read_text(encoding="utf-8").splitlines()]
    coords = {r["id"]: (r["lon"], r["lat"]) for r in (json.loads(x) for x in
              (ROOT / "data/raw/mapillary/images.jsonl").read_text(encoding="utf-8").splitlines())}
    fd = pd.DataFrame(feats)
    fd["lon"] = fd["id"].map(lambda i: coords.get(i, (None, None))[0])
    fd["lat"] = fd["id"].map(lambda i: coords.get(i, (None, None))[1])
    fd = fd.dropna(subset=["lon", "lat"])
    fg = gpd.GeoDataFrame(fd, geometry=gpd.points_from_xy(fd.lon, fd.lat), crs="EPSG:4326").to_crs(METRIC)
    j = gpd.sjoin_nearest(fg, Em[["seg_id", "geometry"]], max_distance=20)
    cols = ["road", "sidewalk", "building", "pole", "vegetation", "sky"]
    return j.groupby("seg_id")[cols].mean()


def places(Em):
    g = gpd.read_parquet(ROOT / "data/raw/overture/places.parquet").to_crs(METRIC)
    def primary(c):
        if isinstance(c, dict):
            return (c.get("primary") or "")
        try:
            return json.loads(c).get("primary") or ""
        except Exception:
            return ""
    g["cat"] = g["categories"].map(primary).str.lower()
    g["night"] = g["cat"].map(lambda s: any(k in s for k in NIGHT_KW))
    tree = STRtree(g.geometry.values)
    night = g["night"].to_numpy()
    dens, ndens = [], []
    for geom in Em.geometry:
        idx = tree.query(geom.buffer(POI_R))
        dens.append(len(idx))
        ndens.append(int(night[idx].sum()) if len(idx) else 0)
    return np.array(dens), np.array(ndens)


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E["street"] = E["name"].apply(coerce) if "name" in E.columns else ""
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    E["seg_id"] = np.arange(len(E))
    Em = E.to_crs(METRIC)
    mid = Em.geometry.interpolate(0.5, normalized=True)

    feat = gpd.GeoDataFrame({
        "seg_id": E["seg_id"], "u": E["u"], "v": E["v"], "key": E["key"],
        "street": E["street"], "highway": E["highway"], "length_m": E["length"].round(1),
        "lat": mid.to_crs("EPSG:4326").y.values, "lon": mid.to_crs("EPSG:4326").x.values,
    }, geometry=Em.geometry.values, crs=METRIC)

    print("lighting ..."); feat["light"] = lighting.lighting_scores(Em, lamps())
    print("enclosure ..."); feat["encl_frontage"], feat["encl_height"] = enclosure(Em, gpd.read_parquet(ROOT / "data/raw/overture/buildings.parquet").to_crs(METRIC))
    print("businesses ..."); feat["poi_density"], feat["poi_night_density"] = places(Em)
    print("visual ...")
    vis = visual(Em)
    for c in ["road", "sidewalk", "building", "pole", "vegetation", "sky"]:
        feat["vis_" + c] = feat["seg_id"].map(vis[c]) if c in vis else np.nan
    feat["has_visual"] = feat["seg_id"].isin(vis.index)

    feat.to_file(OUT / "segment_features.gpkg", driver="GPKG")
    pd.DataFrame(feat.drop(columns="geometry")).to_parquet(OUT / "segment_features.parquet")

    cov = {c: f"{feat[c].notna().mean():.0%}" for c in
           ["light", "encl_frontage", "poi_density", "vis_sidewalk"]}
    print(f"\nsegments: {len(feat)} | feature coverage: {cov}")
    print(f"with visual: {int(feat['has_visual'].sum())} | "
          f"median poi_density: {int(feat['poi_density'].median())} | "
          f"median night POIs: {int(feat['poi_night_density'].median())}")
    print("saved segment_features.gpkg + .parquet")


if __name__ == "__main__":
    main()

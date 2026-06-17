"""Label-free experiment: can daytime imagery recover the lighting signal?

Simulate-missingness premise: hide the streetlight inventory on held-out blocks and try
to predict their (structured) lighting potential from imagery-derived visual features
alone. Spatial block cross-validation (grid cells as groups) so neighbours do not leak.

This predicts STRUCTURED lighting potential, not measured night illumination (no night
labels yet). A positive result means imagery carries lighting-relevant signal and could
gap-fill where no inventory exists; a weak result means the inventory is irreplaceable.

Run:  python scripts/exp_imagery_predicts_lighting.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from shapely.geometry import box  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from waylit import arcgis, lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
ORG = CFG["arcgis_org_base"]
METRIC = "EPSG:32611"
OUT = ROOT / "data" / "processed"
FEATS = ["road", "sidewalk", "building", "pole", "vegetation", "sky"]
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}


def coerce(v):
    return (v[0] if v else "") if isinstance(v, list) else ("" if v is None else str(v))


def lamps_gdf():
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    bb = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    feats, off = [], 0
    while True:
        page = arcgis.query(sl, where="Retired_Date IS NULL", outFields="Wattage,Height",
                            geometry=bb, geometryType="esriGeometryEnvelope", inSR="4326",
                            spatialRel="esriSpatialRelIntersects", outSR="4326",
                            returnGeometry="true", resultOffset=off,
                            resultRecordCount=2000).get("features", [])
        feats += page
        if len(page) < 2000:
            break
        off += 2000
    r = [{"Wattage": f["attributes"].get("Wattage"), "Height": f["attributes"].get("Height"),
          "x": f["geometry"]["x"], "y": f["geometry"]["y"]} for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(pd.DataFrame(r),
                         geometry=gpd.points_from_xy([d["x"] for d in r], [d["y"] for d in r]),
                         crs="EPSG:4326").to_crs(METRIC)
    g["Height"] = g["Height"].astype(float) * 0.3048
    return g


def cv_eval(X, y, groups, label):
    pred = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    rho = spearmanr(pred, y).statistic
    print(f"  [{label}] held-out Spearman {rho:.3f} | R2 {r2_score(y, pred):.3f} "
          f"| MAE {mean_absolute_error(y, pred):.3f}")
    return {"label": label, "spearman": round(float(rho), 3),
            "r2": round(float(r2_score(y, pred)), 3),
            "mae": round(float(mean_absolute_error(y, pred)), 3)}


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    E["seg_id"] = np.arange(len(E))
    Em = E.to_crs(METRIC)
    Em["light"] = lighting.lighting_scores(Em, lamps_gdf())

    # re-aggregate cached per-image CV features to these same segments
    coords = {r["id"]: (r["lon"], r["lat"]) for r in
              (json.loads(x) for x in (ROOT / "data/raw/mapillary/images.jsonl")
               .read_text(encoding="utf-8").splitlines())}
    feats = [json.loads(x) for x in (OUT / "visual_features.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    fd = pd.DataFrame(feats)
    fd["lon"] = fd["id"].map(lambda i: coords.get(i, (None, None))[0])
    fd["lat"] = fd["id"].map(lambda i: coords.get(i, (None, None))[1])
    fd = fd.dropna(subset=["lon", "lat"])
    fg = gpd.GeoDataFrame(fd, geometry=gpd.points_from_xy(fd.lon, fd.lat),
                          crs="EPSG:4326").to_crs(METRIC)
    j = gpd.sjoin_nearest(fg, Em[["seg_id", "geometry"]], max_distance=20)
    vis = j.groupby("seg_id")[FEATS].mean()

    df = Em[["seg_id", "light", "geometry"]].merge(vis, on="seg_id", how="inner")
    cen = df.set_geometry("geometry").geometry.centroid
    xmin, ymin, xmax, ymax = df.set_geometry("geometry").total_bounds
    cell = (np.clip(((cen.x - xmin) / ((xmax - xmin) / 5)).astype(int), 0, 4).astype(str) + ","
            + np.clip(((cen.y - ymin) / ((ymax - ymin) / 5)).astype(int), 0, 4).astype(str))

    y = np.log1p(df["light"].to_numpy())
    groups = cell.to_numpy()
    print(f"segments with imagery + lighting: {len(df)} | spatial cells: {len(set(groups))}")
    print("univariate Spearman of each visual feature vs lighting:")
    for f in FEATS:
        print(f"   {f:11s} {spearmanr(df[f], df['light']).statistic:+.3f}")

    results = [cv_eval(df[FEATS].to_numpy(), y, groups, "all visual features"),
               cv_eval(df[[f for f in FEATS if f != 'pole']].to_numpy(), y, groups, "without pole")]
    (OUT / "exp_imagery_predicts_lighting.json").write_text(json.dumps(
        {"n_segments": int(len(df)), "results": results}, indent=2))


if __name__ == "__main__":
    main()

"""Empirical test: does the ENVIRONMENT explain the night-pedestrian crime gradient,
and does poverty add anything beyond it?

Crime is used here ONLY as validation, never as a routing input. We isolate the fair
target: night-time, outdoor, person-crimes (the lone-pedestrian risk). Then:
  1. Do environmental features track that crime gradient? (Spearman)
  2. Does the user's claim hold: poorer tracts -> more night crime? (Spearman)
  3. Spatial cross-validated R2 predicting crime from env-only / poverty-only / env+poverty.
     Does poverty add predictive power beyond environment?
  4. Bias audit: do our environmental features themselves track neighborhood poverty?

Outputs: data/processed/exp_crime_vs_environment.json + seg_env_crime_income.parquet
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import Point, Polygon, box  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
import yaml  # noqa: E402

from waylit import arcgis, lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
ORG = CFG["arcgis_org_base"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
ENV = "esriGeometryEnvelope"
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}
ENVCOLS = ["light", "encl_frontage", "encl_height"]


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


def crime_points():
    cl = f"{ORG}/BPD_Crimes_Public/FeatureServer/0"
    where = ("CrimeType='Person' AND LocationScene IN "
             "('Highway/Road/Alley','Park/Playground','Parking Lot/Garage') "
             "AND Occurred_HR IN (20,21,22,23,0,1,2,3,4,5)")
    feats, off = [], 0
    while True:
        p = arcgis.query(cl, where=where, geometry=BBOX, geometryType=ENV, inSR="4326",
                        spatialRel="esriSpatialRelIntersects", outFields="OBJECTID",
                        returnGeometry="true", outSR="4326", resultOffset=off,
                        resultRecordCount=2000).get("features", [])
        feats += p
        if len(p) < 2000:
            break
        off += 2000
    pts = [(f["geometry"]["x"], f["geometry"]["y"]) for f in feats if f.get("geometry")]
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(
        [x for x, _ in pts], [y for _, y in pts]), crs="EPSG:4326").to_crs(METRIC)


def poverty_tracts():
    esri = ("https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
            "ACS_Household_Income_Distribution_Boundaries/FeatureServer/2")
    d = arcgis.query(esri, where="1=1", geometry=BBOX, geometryType=ENV, inSR="4326",
                     spatialRel="esriSpatialRelIntersects",
                     outFields="B19001_calc_pctLT75E", returnGeometry="true", outSR="4326")
    feats = d.get("features", [])
    if not feats:
        raise RuntimeError("no poverty features")
    polys, vals = [], []
    for f in feats:
        g = f.get("geometry")
        if not g or "rings" not in g:
            continue
        rings = g["rings"]
        polys.append(Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0]))
        vals.append(f["attributes"].get("B19001_calc_pctLT75E"))
    gdf = gpd.GeoDataFrame({"low_income_share": vals}, geometry=polys, crs="EPSG:4326").to_crs(METRIC)
    gdf["low_income_share"] = pd.to_numeric(gdf["low_income_share"], errors="coerce")
    return gdf.dropna(subset=["low_income_share"])


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
                hv = h[idx[0]]
                if not np.isnan(hv):
                    hs.append(hv)
        front.append(hit / len(pts) if pts else 0.0)
        mh.append(float(np.mean(hs)) if hs else 0.0)
    return np.array(front), np.array(mh)


def cv_r2(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return round(float(r2_score(y, pred)), 3)


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    Em = E.to_crs(METRIC)
    mid = Em.geometry.interpolate(0.5, normalized=True)
    df = pd.DataFrame({"light": lighting.lighting_scores(Em, lamps()),
                       "mx": mid.x.to_numpy(), "my": mid.y.to_numpy()})
    bld = gpd.read_parquet(ROOT / "data/raw/overture/buildings.parquet").to_crs(METRIC)
    df["encl_frontage"], df["encl_height"] = enclosure(Em, bld)

    cp = crime_points()
    ctree = cKDTree(np.c_[cp.geometry.x.to_numpy(), cp.geometry.y.to_numpy()])
    df["crime"] = [len(ctree.query_ball_point((x, y), 100.0)) for x, y in zip(df["mx"], df["my"])]
    print(f"segments: {len(df)} | night/outdoor/person incidents: {len(cp)} | "
          f"segments with >=1 nearby: {(df['crime'] > 0).mean():.0%}")

    income_ok = True
    try:
        tr = poverty_tracts()
        segpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["mx"], df["my"]), crs=METRIC)
        j = gpd.sjoin(segpts, tr[["low_income_share", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        df["low_income_share"] = j["low_income_share"].to_numpy()
        df = df.dropna(subset=["low_income_share"]).reset_index(drop=True)
        print(f"segments with poverty data: {len(df)} | tracts: {len(tr)}")
    except Exception as e:
        income_ok = False
        print("poverty data unavailable:", str(e)[:160])

    df.to_parquet(ROOT / "data/processed/seg_env_crime_income.parquet")

    print("\nSpearman of night crime vs each environmental feature:")
    for c in ENVCOLS:
        print(f"   {c:14s} {spearmanr(df[c], df['crime']).statistic:+.3f}")

    cx = np.clip(((df["mx"] - df["mx"].min()) / ((df["mx"].max() - df["mx"].min()) / 5)).astype(int), 0, 4)
    cy = np.clip(((df["my"] - df["my"].min()) / ((df["my"].max() - df["my"].min()) / 5)).astype(int), 0, 4)
    groups = (cx.astype(str) + "," + cy.astype(str)).to_numpy()
    y = np.log1p(df["crime"].to_numpy())

    out = {"n_segments": int(len(df)), "n_incidents": int(len(cp)),
           "spearman_env_vs_crime": {c: round(float(spearmanr(df[c], df["crime"]).statistic), 3) for c in ENVCOLS},
           "cv_r2_env_only": cv_r2(df[ENVCOLS].to_numpy(), y, groups)}
    print(f"\nspatial-CV R2 predicting night crime  env-only: {out['cv_r2_env_only']}")
    if income_ok:
        li = "low_income_share"
        out["spearman_crime_vs_low_income"] = round(float(spearmanr(df["crime"], df[li]).statistic), 3)
        out["cv_r2_income_only"] = cv_r2(df[[li]].to_numpy(), y, groups)
        out["cv_r2_env_plus_income"] = cv_r2(df[ENVCOLS + [li]].to_numpy(), y, groups)
        out["income_gain_over_env"] = round(out["cv_r2_env_plus_income"] - out["cv_r2_env_only"], 3)
        out["bias_spearman_light_vs_low_income"] = round(float(spearmanr(df["light"], df[li]).statistic), 3)
        print(f"user's claim: Spearman(night crime, low-income share) = {out['spearman_crime_vs_low_income']:+.3f}")
        print(f"spatial-CV R2: income-only {out['cv_r2_income_only']} | env+income {out['cv_r2_env_plus_income']} "
              f"(poverty adds {out['income_gain_over_env']:+.3f} over env)")
        print(f"bias audit: Spearman(lighting, low-income share) = {out['bias_spearman_light_vs_low_income']:+.3f}")
    (ROOT / "data/processed/exp_crime_vs_environment.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Finish the crime test: join tract poverty to the saved env+crime table and run the
poverty comparisons. Reuses data/processed/seg_env_crime_income.parquet (no recompute).
"""
import json
import pathlib
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
B = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
ENVCOLS = ["light", "encl_frontage", "encl_height"]


def fetch_tracts():
    base = ("https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
            "ACS_Household_Income_Distribution_Boundaries/FeatureServer/2/query")
    params = dict(where="1=1", geometry=BBOX, geometryType="esriGeometryEnvelope", inSR="4326",
                  spatialRel="esriSpatialRelIntersects", outFields="B19001_calc_pctLT75E",
                  returnGeometry="true", outSR="4326", maxAllowableOffset="15",
                  geometryPrecision="5", f="json")
    url = base + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def cv_r2(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return round(float(r2_score(y, pred)), 3)


def main():
    df = pd.read_parquet(ROOT / "data/processed/seg_env_crime_income.parquet")
    d = fetch_tracts()
    polys, vals = [], []
    for f in d.get("features", []):
        g = f.get("geometry")
        if not g or "rings" not in g:
            continue
        rings = g["rings"]
        polys.append(Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0]))
        vals.append(f["attributes"].get("B19001_calc_pctLT75E"))
    tr = gpd.GeoDataFrame({"low_income_share": pd.to_numeric(vals, errors="coerce")},
                          geometry=polys, crs="EPSG:4326").to_crs(METRIC).dropna(subset=["low_income_share"])
    print(f"tracts: {len(tr)} | low-income share range {tr.low_income_share.min():.0f}-{tr.low_income_share.max():.0f}%")

    segpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["mx"], df["my"]), crs=METRIC)
    j = gpd.sjoin(segpts, tr[["low_income_share", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    df["low_income_share"] = j["low_income_share"].to_numpy()
    df = df.dropna(subset=["low_income_share"]).reset_index(drop=True)
    print(f"segments with poverty data: {len(df)}")

    cx = np.clip(((df["mx"] - df["mx"].min()) / ((df["mx"].max() - df["mx"].min()) / 5)).astype(int), 0, 4)
    cy = np.clip(((df["my"] - df["my"].min()) / ((df["my"].max() - df["my"].min()) / 5)).astype(int), 0, 4)
    groups = (cx.astype(str) + "," + cy.astype(str)).to_numpy()
    y = np.log1p(df["crime"].to_numpy())
    li = "low_income_share"

    res = {
        "n_segments": int(len(df)),
        "spearman_crime_vs_low_income": round(float(spearmanr(df["crime"], df[li]).statistic), 3),
        "cv_r2_env_only": cv_r2(df[ENVCOLS].to_numpy(), y, groups),
        "cv_r2_income_only": cv_r2(df[[li]].to_numpy(), y, groups),
        "cv_r2_env_plus_income": cv_r2(df[ENVCOLS + [li]].to_numpy(), y, groups),
        "bias_spearman_light_vs_low_income": round(float(spearmanr(df["light"], df[li]).statistic), 3),
        "bias_spearman_encl_height_vs_low_income": round(float(spearmanr(df["encl_height"], df[li]).statistic), 3),
    }
    res["income_gain_over_env"] = round(res["cv_r2_env_plus_income"] - res["cv_r2_env_only"], 3)
    print(json.dumps(res, indent=2))
    out = ROOT / "data/processed/exp_crime_vs_environment.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(res)
    out.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()

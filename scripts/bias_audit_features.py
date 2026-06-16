"""Bias audit: does each environmental feature track neighborhood poverty?

For every feature in segment_features, compute its Spearman correlation with tract-level
low-income share. Features that strongly track poverty are flagged as potential
demographic proxies (to drop or handle carefully before they enter any ranking).

Output: data/processed/bias_audit.json
"""
import json
import pathlib
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import pandas as pd
import yaml
from scipy.stats import spearmanr
from shapely.geometry import Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
B = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
FEATURES = ["light", "encl_frontage", "encl_height", "poi_density", "poi_night_density",
            "vis_sidewalk", "vis_vegetation", "vis_sky", "vis_building", "vis_pole"]


def fetch_tracts():
    base = ("https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
            "ACS_Household_Income_Distribution_Boundaries/FeatureServer/2/query")
    params = dict(where="1=1", geometry=BBOX, geometryType="esriGeometryEnvelope", inSR="4326",
                  spatialRel="esriSpatialRelIntersects", outFields="B19001_calc_pctLT75E",
                  returnGeometry="true", outSR="4326", maxAllowableOffset="15",
                  geometryPrecision="5", f="json")
    url = base + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            if d.get("features"):
                return d
            last = {"empty_or_error": {k: d[k] for k in d if k != "fields"}}
        except Exception as e:
            last = {"exc": str(e)[:200]}
        time.sleep(4)
    raise RuntimeError(f"tract fetch failed: {json.dumps(last)[:300]}")


def main():
    feat = gpd.read_file(ROOT / "data/processed/segment_features.gpkg")
    pts = gpd.GeoDataFrame(feat, geometry=gpd.points_from_xy(feat["lon"], feat["lat"]),
                           crs="EPSG:4326").to_crs(METRIC)

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

    j = gpd.sjoin(pts, tr[["low_income_share", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]

    def flag(r):
        a = abs(r)
        return "WATCH" if a >= 0.4 else ("mild" if a >= 0.2 else "clean")

    rows = []
    for c in FEATURES:
        sub = j[[c, "low_income_share"]].dropna()
        if len(sub) < 30:
            continue
        rho = float(spearmanr(sub[c], sub["low_income_share"]).statistic)
        rows.append({"feature": c, "spearman_vs_poverty": round(rho, 3), "n": len(sub), "flag": flag(rho)})
    rows.sort(key=lambda x: -abs(x["spearman_vs_poverty"]))

    print(f"tracts: {len(tr)} | segments joined: {j['low_income_share'].notna().sum()}\n")
    print(f"{'feature':18s}{'rho vs poverty':>16s}{'   flag':>8s}")
    for r in rows:
        print(f"{r['feature']:18s}{r['spearman_vs_poverty']:>16.3f}   {r['flag']}")
    (ROOT / "data/processed/bias_audit.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

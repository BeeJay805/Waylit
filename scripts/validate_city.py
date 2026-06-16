"""Validate environmental comfort in a high-variance city: does it track night-pedestrian
crime, and is it demographically clean?

Crime and demographics are HELD-OUT targets, NEVER features (the project's rule). This mirrors
the Boise exp_crime_vs_environment methodology and adds (a) a race axis to the bias audit and
(b) a transfer test: apply the Boise-learned comfort weights to this city and see whether they
predict lower crime without keying on demographics. Config-driven via config/cities/<city>.yaml.

Honest framing: a city with real danger variance is where 'does comfort mean anything' and
'is comfort a demographic proxy' can actually be tested. Crime here conflates exposure (busy
places have more crime AND more comfort features), so we report it with that confound stated.

Run:   python scripts/validate_city.py la
Probe: python scripts/validate_city.py la --probe      # test data fetches only, no heavy loops
"""
import argparse
import io
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # import build_city_features

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import box  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from waylit import arcgis  # noqa: E402
from waylit import lighting as L  # noqa: E402
from build_city_features import overture, enclosure, coerce, WALKABLE, NIGHT_KW  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
# Boise-learned comfort directions, for the transfer test (signs only; magnitudes differ in units)
BOISE_SIGN = {"lit_frac": +1, "encl_frontage": -1, "encl_height": +1,
              "poi_density": -1, "poi_night_density": +1}
FEATS = ["lit_frac", "encl_frontage", "encl_height", "poi_density", "poi_night_density"]
UA = {"User-Agent": "waylit-research/0.1"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


# ---------- segments + features ----------
def segments(B, METRIC):
    G = ox.convert.to_undirected(ox.graph_from_polygon(
        box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"]),
        network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    Em = E.to_crs(METRIC)
    mid = Em.geometry.interpolate(0.5, normalized=True)
    return Em, mid.x.to_numpy(), mid.y.to_numpy()


def lamps(service, B, METRIC, cache):
    if cache.exists():
        return gpd.read_parquet(cache)
    bbox = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    feats, off = [], 0
    while True:
        p = arcgis.query(service, where="1=1", geometry=bbox, geometryType="esriGeometryEnvelope",
                         inSR="4326", spatialRel="esriSpatialRelIntersects", outFields="OBJECTID",
                         returnGeometry="true", outSR="4326", resultOffset=off,
                         resultRecordCount=5000).get("features", [])
        feats += p
        if len(p) < 5000:
            break
        off += 5000
    xy = [(f["geometry"]["x"], f["geometry"]["y"]) for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x for x, _ in xy], [y for _, y in xy]),
                         crs="EPSG:4326").to_crs(METRIC)
    g.to_parquet(cache)
    return g


def lit_fraction(Em, lamp_gdf, reach=25.0, step=18.0):
    """Lamp-coverage proxy: fraction of points sampled along a segment that have a streetlight
    within `reach` metres. Bounded 0..1, analogous to enclosure frontage. Not photometric."""
    tree = cKDTree(np.c_[lamp_gdf.geometry.x.to_numpy(), lamp_gdf.geometry.y.to_numpy()])
    out = []
    for geom in Em.geometry:
        pts = L._sample_xy(geom, step=step)
        if not pts:
            out.append(0.0)
            continue
        hit = sum(1 for px, py in pts if tree.query_ball_point((px, py), reach))
        out.append(hit / len(pts))
    return np.array(out)


def places_density(Em, pl):
    def primary(c):
        if isinstance(c, dict):
            return c.get("primary") or ""
        try:
            return json.loads(c).get("primary") or ""
        except Exception:
            return ""
    pl = pl.copy()
    pl["cat"] = pl["categories"].map(primary).str.lower()
    pl["night"] = pl["cat"].map(lambda s: any(k in s for k in NIGHT_KW))
    tree = STRtree(pl.geometry.values)
    night = pl["night"].to_numpy()
    dens, ndens = [], []
    for geom in Em.geometry:
        idx = tree.query(geom.buffer(50))
        dens.append(len(idx))
        ndens.append(int(night[idx].sum()) if len(idx) else 0)
    return np.array(dens), np.array(ndens)


# ---------- held-out targets ----------
def crime_socrata(url, B, METRIC, cache):
    if cache.exists():
        return gpd.read_parquet(cache)
    where = (
        f"lat >= {B['min_lat']} AND lat <= {B['max_lat']} AND "
        f"lon >= {B['min_lon']} AND lon <= {B['max_lon']} AND lat != 0 "
        "AND (time_occ >= '2000' OR time_occ <= '0559') "
        "AND (upper(crm_cd_desc) like '%ROBBERY%' OR upper(crm_cd_desc) like '%ASSAULT%' "
        "OR upper(crm_cd_desc) like '%BATTERY%') "
        "AND (upper(premis_desc) like '%STREET%' OR upper(premis_desc) like '%SIDEWALK%' "
        "OR upper(premis_desc) like '%ALLEY%' OR upper(premis_desc) like '%PARK%')")
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode({"$select": "lat,lon", "$where": where,
                                    "$limit": 50000, "$offset": off})
        page = get_json(url + "?" + q)
        rows += page
        if len(page) < 50000:
            break
        off += 50000
    df = pd.DataFrame(rows).astype({"lat": float, "lon": float})
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                         crs="EPSG:4326").to_crs(METRIC)
    g.to_parquet(cache)
    return g


def acs_blockgroups(cfg, METRIC, cache):
    if cache.exists():
        return gpd.read_parquet(cache)
    keyfile = ROOT / "secrets" / "census_api_key.txt"
    key = os.environ.get("CENSUS_API_KEY") or (keyfile.read_text().strip() if keyfile.exists() else "")
    if not key:
        raise RuntimeError("no Census API key: set CENSUS_API_KEY env var or put a free key in "
                           "secrets/census_api_key.txt (https://api.census.gov/data/key_signup.html)")
    yr, st, co = cfg["year"], cfg["state"], cfg["county"]
    api = (f"https://api.census.gov/data/{yr}/acs/acs5?get=B19013_001E,B02001_001E,"
           f"B02001_002E&for=block%20group:*&in=state:{st}%20county:{co}%20tract:*&key={key}")
    rows = get_json(api)
    hdr, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=hdr)
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    df.loc[df["income"] < 0, "income"] = np.nan
    tot = pd.to_numeric(df["B02001_001E"], errors="coerce")
    white = pd.to_numeric(df["B02001_002E"], errors="coerce")
    df["pct_nonwhite"] = np.where(tot > 0, 1 - white / tot, np.nan)

    tiger = RAW / "census" / f"cb_{yr}_{st}_bg_500k.zip"
    tiger.parent.mkdir(parents=True, exist_ok=True)
    if not tiger.exists():
        turl = f"https://www2.census.gov/geo/tiger/GENZ{yr}/shp/cb_{yr}_{st}_bg_500k.zip"
        with urllib.request.urlopen(urllib.request.Request(turl, headers=UA), timeout=180) as r:
            tiger.write_bytes(r.read())
    geo = gpd.read_file(f"zip://{tiger}")
    geo = geo[geo["COUNTYFP"] == co][["GEOID", "geometry"]]
    bg = geo.merge(df[["GEOID", "income", "pct_nonwhite"]], on="GEOID", how="left").to_crs(METRIC)
    bg.to_parquet(cache)
    return bg


# ---------- analysis ----------
def cv_r2(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return round(float(r2_score(y, pred)), 3)


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(spearmanr(a[m], b[m]).statistic), 3) if m.sum() > 10 else None


def main(city, probe, rebuild):
    cfg = yaml.safe_load((ROOT / "config" / "cities" / f"{city}.yaml").read_text())
    B, METRIC = cfg["bbox"], cfg["crs_metric"]
    bbox_str = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    print(f"=== {cfg['name']} ===")

    if probe:
        lp = lamps(cfg["streetlight_service"], B, METRIC, RAW / f"{city}_lamps.parquet")
        print(f"streetlights: {len(lp)}")
        cp = crime_socrata(cfg["crime_socrata"], B, METRIC, RAW / f"{city}_crime.parquet")
        print(f"night/outdoor/person crime incidents: {len(cp)}")
        try:
            bg = acs_blockgroups(cfg["census"], METRIC, RAW / f"{city}_acs_bg.parquet")
            print(f"ACS block groups (county): {len(bg)} | with income: {bg['income'].notna().sum()}")
        except Exception as e:
            print("ACS (demographics) needs a key:", str(e)[:160])
        return

    cp = crime_socrata(cfg["crime_socrata"], B, METRIC, RAW / f"{city}_crime.parquet")
    featcache = PROC / f"{city}_seg_features.parquet"
    if featcache.exists() and not rebuild:
        df = pd.read_parquet(featcache)
        print(f"loaded cached feature table: {len(df)} segments (pass --rebuild to recompute)")
    else:
        Em, mx, my = segments(B, METRIC)
        print(f"walk segments: {len(Em)}")
        bld = overture(city, "building", bbox_str, METRIC)
        pl = overture(city, "place", bbox_str, METRIC)
        print(f"Overture: {len(bld)} buildings, {len(pl)} places")
        df = pd.DataFrame({"seg_id": np.arange(len(Em)), "mx": mx, "my": my})
        print("enclosure ..."); df["encl_frontage"], df["encl_height"] = enclosure(Em, bld)
        print("business ..."); df["poi_density"], df["poi_night_density"] = places_density(Em, pl)
        print("lighting (lamp coverage) ...")
        df["lit_frac"] = lit_fraction(Em, lamps(cfg["streetlight_service"], B, METRIC,
                                                RAW / f"{city}_lamps.parquet"))
        ctree = cKDTree(np.c_[cp.geometry.x.to_numpy(), cp.geometry.y.to_numpy()])
        df["crime"] = [len(ctree.query_ball_point((x, y), 100.0)) for x, y in zip(mx, my)]
        df.to_parquet(featcache)
    mx, my = df["mx"].to_numpy(), df["my"].to_numpy()
    print(f"crime incidents: {len(cp)} | segments with >=1 within 100m: {(df['crime'] > 0).mean():.0%}")

    demo_ok = True
    try:
        bg = acs_blockgroups(cfg["census"], METRIC, RAW / f"{city}_acs_bg.parquet")
        seg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(mx, my), crs=METRIC)
        j = gpd.sjoin(seg, bg[["income", "pct_nonwhite", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        df["income"] = j["income"].to_numpy()
        df["pct_nonwhite"] = j["pct_nonwhite"].to_numpy()
        print(f"demographics joined | segs with income: {df['income'].notna().sum()} | "
              f"income range ${np.nanpercentile(df['income'],5):,.0f}-${np.nanpercentile(df['income'],95):,.0f}")
    except Exception as e:
        demo_ok = False
        print("demographics unavailable:", str(e)[:200])

    df.to_parquet(PROC / f"{city}_seg_features.parquet")

    # spatial blocks for CV (5x5 grid over the bbox)
    cx = np.clip(((df["mx"] - df["mx"].min()) / ((df["mx"].max() - df["mx"].min()) / 5)).astype(int), 0, 4)
    cy = np.clip(((df["my"] - df["my"].min()) / ((df["my"].max() - df["my"].min()) / 5)).astype(int), 0, 4)
    groups = (cx.astype(str) + "," + cy.astype(str)).to_numpy()
    y = np.log1p(df["crime"].to_numpy())

    out = {"city": cfg["name"], "n_segments": int(len(df)), "n_incidents": int(len(cp)),
           "lighting": "lamp-coverage proxy (no photometric attrs)",
           "spearman_feature_vs_night_crime": {c: sp(df[c].to_numpy(dtype=float), df["crime"].to_numpy(dtype=float)) for c in FEATS},
           "cv_r2_env_only": cv_r2(df[FEATS].to_numpy(), y, groups)}

    # transfer test: Boise-sign comfort score (standardized in THIS city), vs crime
    z = np.column_stack([(df[c] - df[c].mean()) / (df[c].std() + 1e-9) * BOISE_SIGN[c] for c in FEATS])
    df["comfort_boise_signs"] = z.sum(axis=1)
    out["comfort_score_vs_crime_spearman"] = sp(df["comfort_boise_signs"].to_numpy(),
                                                 df["crime"].to_numpy(dtype=float))

    if demo_ok:
        inc = df["income"].to_numpy(dtype=float)
        nw = df["pct_nonwhite"].to_numpy(dtype=float)
        m = np.isfinite(inc)
        out["bias_audit_feature_vs_income"] = {c: sp(df[c].to_numpy(dtype=float), inc) for c in FEATS}
        out["bias_audit_feature_vs_pct_nonwhite"] = {c: sp(df[c].to_numpy(dtype=float), nw) for c in FEATS}
        out["bias_comfort_score_vs_income"] = sp(df["comfort_boise_signs"].to_numpy(), inc)
        out["bias_comfort_score_vs_pct_nonwhite"] = sp(df["comfort_boise_signs"].to_numpy(), nw)
        out["crime_vs_income_spearman"] = sp(df["crime"].to_numpy(dtype=float), inc)
        out["crime_vs_pct_nonwhite_spearman"] = sp(df["crime"].to_numpy(dtype=float), nw)
        dfm = df[m].reset_index(drop=True)
        gm = groups[m]
        ym = np.log1p(dfm["crime"].to_numpy())
        demo = dfm[["income", "pct_nonwhite"]].fillna(dfm[["income", "pct_nonwhite"]].median()).to_numpy()
        out["cv_r2_demographics_only"] = cv_r2(demo, ym, gm)
        out["cv_r2_env_only_on_demo_subset"] = cv_r2(dfm[FEATS].to_numpy(), ym, gm)
        out["cv_r2_env_plus_demographics"] = cv_r2(np.column_stack([dfm[FEATS].to_numpy(), demo]), ym, gm)
        out["demographics_gain_over_env"] = round(
            out["cv_r2_env_plus_demographics"] - out["cv_r2_env_only_on_demo_subset"], 3)

    out["caveats"] = [
        "Crime conflates exposure: busy commercial blocks have more incidents AND more comfort "
        "features, so a positive comfort/crime link is partly an activity confound, not proof.",
        "Lighting is a lamp-coverage proxy (LA has no wattage/height), not the Boise photometric score.",
        "Comfort directions are transferred from a single Boise rater's daytime judgments; a null "
        "or biased result is informative, a positive result is preliminary.",
        "Bias audit asks whether features proxy income/race in a diverse city; near-zero is the goal "
        "(Boise baseline was ~0.00 vs poverty, but Boise is not diverse).",
        "Crime + demographics are held-out targets here, never routing inputs.",
    ]
    (PROC / f"{city}_validation.json").write_text(json.dumps(out, indent=2))

    print("\nSpearman, each feature vs night crime:")
    for c, v in out["spearman_feature_vs_night_crime"].items():
        print(f"  {c:18s} {v}")
    print(f"spatial-CV R2 predicting night crime, env-only: {out['cv_r2_env_only']}")
    print(f"comfort score (Boise signs) vs crime: {out['comfort_score_vs_crime_spearman']}")
    if demo_ok:
        print("\nbias audit (|small| = clean):")
        for c in FEATS:
            print(f"  {c:18s} vs income {out['bias_audit_feature_vs_income'][c]:+}  "
                  f"vs %nonwhite {out['bias_audit_feature_vs_pct_nonwhite'][c]:+}")
        print(f"  comfort score      vs income {out['bias_comfort_score_vs_income']:+}  "
              f"vs %nonwhite {out['bias_comfort_score_vs_pct_nonwhite']:+}")
        print(f"\nspatial-CV R2: env {out['cv_r2_env_only_on_demo_subset']} | demo {out['cv_r2_demographics_only']} "
              f"| env+demo {out['cv_r2_env_plus_demographics']} (demo adds {out['demographics_gain_over_env']:+})")
    print(f"\nwrote {city}_validation.json + {city}_seg_features.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    main(a.city, a.probe, a.rebuild)

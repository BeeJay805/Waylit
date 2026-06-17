"""Exposure-normalized crime validation: does comfort (especially lighting) track LOWER night
crime RISK once ambient population is held constant, or does it just track where people are?

Raw incident counts rise with exposure (more people out -> more incidents), and comfort features
also track activity, so comfort correlates POSITIVELY with raw counts (the +0.30 in validate_city).
Here we control for exposure proxies that are INDEPENDENT of the comfort features:
  residential population  (ACS B01003, where people live)
  worker density          (LODES WAC total jobs, daytime activity)
Lighting is the cleanest test because, unlike business density, it is not itself an exposure
measure. This is observational with imperfect night-exposure proxies; no causal claim is made.

Reads cached data/processed/<city>_seg_features.parquet + data/raw/<city>_acs_bg.parquet.
Run:  CENSUS_API_KEY=... python scripts/exposure_normalize.py la
Output: data/processed/<city>_exposure_validation.json
"""
import argparse
import io
import json
import os
import pathlib
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
COMFORT = ["lit_frac", "encl_frontage", "encl_height", "poi_density", "poi_night_density"]
BOISE_SIGN = {"lit_frac": +1, "encl_frontage": -1, "encl_height": +1,
              "poi_density": -1, "poi_night_density": +1}
FIPS_ABBR = {"06": "ca", "16": "id", "49": "ut"}
UA = {"User-Agent": "waylit-research/0.1"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.load(r)


def census_key():
    k = os.environ.get("CENSUS_API_KEY")
    if not k:
        f = ROOT / "secrets" / "census_api_key.txt"
        k = f.read_text().strip() if f.exists() else ""
    if not k:
        raise RuntimeError("set CENSUS_API_KEY or secrets/census_api_key.txt")
    return k


def acs_pop(cen):
    yr, st, co = cen["year"], cen["state"], cen["county"]
    url = (f"https://api.census.gov/data/{yr}/acs/acs5?get=B01003_001E&for=block%20group:*"
           f"&in=state:{st}%20county:{co}%20tract:*&key={census_key()}")
    rows = get_json(url)
    hdr, data = rows[0], rows[1:]
    d = pd.DataFrame(data, columns=hdr)
    d["GEOID"] = d["state"] + d["county"] + d["tract"] + d["block group"]
    d["pop"] = pd.to_numeric(d["B01003_001E"], errors="coerce").clip(lower=0)
    return d.set_index("GEOID")["pop"].to_dict()


def lodes_jobs(st, co, cache):
    """Total jobs per block group from LODES WAC (workplace), summed from block level."""
    if cache.exists():
        j = pd.read_parquet(cache)
        return dict(zip(j["GEOID"], j["jobs"]))
    abbr = FIPS_ABBR.get(st)
    if not abbr:
        return {}
    for yr in (2022, 2021, 2020):
        url = (f"https://lehd.ces.census.gov/data/lodes/LODES8/{abbr}/wac/"
               f"{abbr}_wac_S000_JT00_{yr}.csv.gz")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=240) as r:
                buf = io.BytesIO(r.read())
            break
        except Exception:
            buf = None
    if buf is None:
        return {}
    w = pd.read_csv(buf, compression="gzip", usecols=["w_geocode", "C000"],
                    dtype={"w_geocode": str})
    w = w[w["w_geocode"].str[:5] == st + co]
    w["GEOID"] = w["w_geocode"].str[:12]
    j = w.groupby("GEOID")["C000"].sum().reset_index().rename(columns={"C000": "jobs"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    j.to_parquet(cache)
    return dict(zip(j["GEOID"], j["jobs"]))


def partial_spearman(y, x, Z):
    """Partial Spearman of x with y controlling for the columns of Z (rank + OLS residuals)."""
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    if m.sum() < 30:
        return None
    ry, rx = rankdata(y[m]), rankdata(x[m])
    rZ = np.column_stack([rankdata(Z[m, k]) for k in range(Z.shape[1])])
    ex = rx - LinearRegression().fit(rZ, rx).predict(rZ)
    ey = ry - LinearRegression().fit(rZ, ry).predict(rZ)
    return round(float(np.corrcoef(ex, ey)[0, 1]), 3)


def cv_r2(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return round(float(r2_score(y, pred)), 3)


def main(city):
    cfg = yaml.safe_load((ROOT / "config" / "cities" / f"{city}.yaml").read_text())
    METRIC, cen = cfg["crs_metric"], cfg["census"]
    df = pd.read_parquet(PROC / f"{city}_seg_features.parquet")
    bg = gpd.read_parquet(RAW / f"{city}_acs_bg.parquet").to_crs(METRIC)
    bg["area_km2"] = bg.geometry.area / 1e6

    pop = acs_pop(cen)
    jobs = lodes_jobs(cen["state"], cen["county"], RAW / f"{city}_lodes_wac.parquet")
    bg["pop"] = bg["GEOID"].map(pop)
    bg["jobs"] = bg["GEOID"].map(jobs).fillna(0) if jobs else np.nan
    bg["pop_density"] = bg["pop"] / bg["area_km2"]
    bg["job_density"] = bg["jobs"] / bg["area_km2"]
    have_jobs = bool(jobs)
    print(f"exposure proxies: residential pop{' + LODES jobs' if have_jobs else ' only (LODES unavailable)'}")

    seg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["mx"], df["my"]), crs=METRIC)
    cols = ["pop_density", "job_density", "geometry"] if have_jobs else ["pop_density", "geometry"]
    j = gpd.sjoin(seg, bg[cols], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    df["pop_density"] = j["pop_density"].to_numpy()
    df["job_density"] = j["job_density"].to_numpy() if have_jobs else 0.0
    df["ambient"] = df["pop_density"].fillna(0) + (df["job_density"].fillna(0) if have_jobs else 0)

    df["comfort"] = np.sum([(df[c] - df[c].mean()) / (df[c].std() + 1e-9) * BOISE_SIGN[c]
                            for c in COMFORT], axis=0)
    crime = df["crime"].to_numpy(dtype=float)
    expo_cols = ["pop_density", "job_density"] if have_jobs else ["pop_density"]
    Z = df[expo_cols].to_numpy(dtype=float)

    # crime rate per ambient population (exposure-normalized incidence)
    df["crime_rate"] = crime / (df["ambient"].to_numpy() + 1.0)

    out = {
        "city": cfg["name"], "n_segments": int(len(df)),
        "exposure_proxies": expo_cols,
        "note": "Does comfort track lower crime RISK after controlling for ambient population? "
                "Lighting is the cleanest test (not itself an exposure measure).",
        "raw_spearman_vs_crime": {c: round(float(spearmanr(df[c], crime).statistic), 3) for c in COMFORT},
        "spearman_vs_crime_rate": {c: round(float(spearmanr(df[c], df["crime_rate"]).statistic), 3) for c in COMFORT},
        "partial_spearman_vs_crime_controlling_exposure": {
            c: partial_spearman(crime, df[c].to_numpy(dtype=float), Z) for c in COMFORT},
        "comfort_score_partial_vs_crime": partial_spearman(crime, df["comfort"].to_numpy(), Z),
    }

    # transparent linear model: standardized, crime(log) ~ exposure + comfort; read the signs
    feats = expo_cols + COMFORT
    X = np.column_stack([(df[c] - df[c].mean()) / (df[c].std() + 1e-9) for c in feats])
    ylog = np.log1p(crime)
    mfit = np.all(np.isfinite(X), axis=1)
    lr = LinearRegression().fit(X[mfit], ylog[mfit])
    out["linear_coeffs_crime_log"] = {f: round(float(c), 3) for f, c in zip(feats, lr.coef_)}

    # spatial-block CV: does comfort add predictive power BEYOND exposure?
    mx, my = df["mx"].to_numpy(), df["my"].to_numpy()
    cx = np.clip(((mx - mx.min()) / ((mx.max() - mx.min()) / 5)).astype(int), 0, 4)
    cy = np.clip(((my - my.min()) / ((my.max() - my.min()) / 5)).astype(int), 0, 4)
    groups = np.array([f"{a},{b}" for a, b in zip(cx, cy)])
    sub = mfit
    out["cv_r2_exposure_only"] = cv_r2(df.loc[sub, expo_cols].to_numpy(), ylog[sub], groups[sub])
    out["cv_r2_exposure_plus_comfort"] = cv_r2(
        df.loc[sub, expo_cols + COMFORT].to_numpy(), ylog[sub], groups[sub])
    out["comfort_gain_over_exposure"] = round(
        out["cv_r2_exposure_plus_comfort"] - out["cv_r2_exposure_only"], 3)

    lit_partial = out["partial_spearman_vs_crime_controlling_exposure"]["lit_frac"]
    lit_lin = out["linear_coeffs_crime_log"]["lit_frac"]
    if lit_partial is not None and lit_partial < -0.03 and lit_lin < 0:
        verdict = (f"After controlling for ambient population, MORE lighting is associated with "
                   f"LESS night crime (partial Spearman {lit_partial}, linear coef {lit_lin}). "
                   "First signal that comfort tracks lower risk, not just lower exposure.")
    elif lit_partial is not None and lit_partial > 0.03:
        verdict = (f"Even after exposure control, lighting still tracks MORE crime (partial "
                   f"{lit_partial}). Either the exposure proxies miss night foot traffic, or "
                   "lit blocks genuinely co-locate with incidents here.")
    else:
        verdict = (f"Lighting's exposure-controlled link to crime is ~flat (partial {lit_partial}, "
                   f"coef {lit_lin}): no clear risk signal either way at this resolution.")
    out["verdict"] = verdict
    out["caveats"] = [
        "Residential pop + worker jobs are imperfect proxies for NIGHT pedestrian exposure.",
        "Observational and ecological (segment/block level); no causal or individual-risk claim.",
        "Lighting is a lamp-coverage proxy (LA has no wattage/height).",
        "Crime stays a held-out target here, never a routing input.",
    ]
    (PROC / f"{city}_exposure_validation.json").write_text(json.dumps(out, indent=2))

    print(f"\nexposure-controlled (partial Spearman vs crime | ambient population):")
    for c in COMFORT:
        print(f"  {c:18s} raw {out['raw_spearman_vs_crime'][c]:+}  ->  controlled "
              f"{out['partial_spearman_vs_crime_controlling_exposure'][c]}")
    print(f"  comfort score      controlled {out['comfort_score_partial_vs_crime']}")
    print(f"\nlinear coef (crime_log, standardized): "
          f"lit {out['linear_coeffs_crime_log']['lit_frac']:+}  "
          f"pop {out['linear_coeffs_crime_log']['pop_density']:+}"
          + (f"  jobs {out['linear_coeffs_crime_log']['job_density']:+}" if have_jobs else ""))
    print(f"spatial-CV R2: exposure {out['cv_r2_exposure_only']} -> +comfort "
          f"{out['cv_r2_exposure_plus_comfort']} (comfort adds {out['comfort_gain_over_exposure']:+})")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {city}_exposure_validation.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    a = ap.parse_args()
    main(a.city)

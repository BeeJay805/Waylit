"""Build the night-pilot field kit.

Selects ~40 downtown segments spanning bright/dim/dark, fixture types, and spatial
spread (preferring blocks that have imagery, with some that do not), and writes:
  data/ground_truth/pilot_segments.csv   ready-to-fill field sheet (no predictions shown)
  data/ground_truth/pilot_map.png        numbered route map
  data/ground_truth/pilot_selection.json selection rationale (with predicted light, for us)

Run:  python scripts/build_pilot_kit.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import box  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from waylit import arcgis, lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
ORG = CFG["arcgis_org_base"]
METRIC = "EPSG:32611"
N_TARGET = 40
WALKABLE = {"residential", "living_street", "tertiary", "secondary", "primary",
            "unclassified", "footway", "pedestrian", "path"}
OUT = ROOT / "data" / "ground_truth"
OUT.mkdir(parents=True, exist_ok=True)


def lamps_gdf():
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    bbox = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    feats, off = [], 0
    while True:
        page = arcgis.query(
            sl, where="Retired_Date IS NULL", outFields="Wattage,Height,Fixture_Type",
            geometry=bbox, geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outSR="4326", returnGeometry="true",
            resultOffset=off, resultRecordCount=2000).get("features", [])
        feats += page
        if len(page) < 2000:
            break
        off += 2000
    rows = [{"Wattage": f["attributes"].get("Wattage"),
             "Height": f["attributes"].get("Height"),
             "Fixture_Type": f["attributes"].get("Fixture_Type"),
             "x": f["geometry"]["x"], "y": f["geometry"]["y"]}
            for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(pd.DataFrame(rows),
                         geometry=gpd.points_from_xy([r["x"] for r in rows],
                                                     [r["y"] for r in rows]),
                         crs="EPSG:4326").to_crs(METRIC)
    g["Height"] = g["Height"].astype(float) * 0.3048
    return g


def coerce(v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.convert.to_undirected(
        ox.graph_from_polygon(poly, network_type="walk", retain_all=True))
    E = ox.graph_to_gdfs(G, nodes=False).reset_index()
    E["highway"] = E["highway"].apply(coerce)
    E["street"] = E["name"].apply(coerce) if "name" in E.columns else ""
    E = E[E["highway"].isin(WALKABLE) & (E["length"] >= 25)].reset_index(drop=True)
    E["seg_id"] = np.arange(len(E))
    Em = E.to_crs(METRIC)

    lamps = lamps_gdf()
    E["light"] = lighting.lighting_scores(Em, lamps)

    # clean segment-level imagery coverage (undirected; fixes the earlier artifact)
    rows = [json.loads(x) for x in (ROOT / "data" / "raw" / "mapillary" / "images.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    idf = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    imgs = gpd.GeoDataFrame(idf, geometry=gpd.points_from_xy(idf.lon, idf.lat),
                            crs="EPSG:4326").to_crs(METRIC)
    seg_geom = Em[["seg_id", "geometry"]]
    jn = gpd.sjoin_nearest(imgs, seg_geom, max_distance=20, distance_col="d")
    E["n_img"] = E["seg_id"].map(jn.groupby("seg_id").size()).fillna(0).astype(int)
    cov = round(100 * (E["n_img"] > 0).mean(), 1)

    # nearest-lamp fixture type per segment (context only, not a prediction)
    fx = gpd.sjoin_nearest(seg_geom, lamps[["Fixture_Type", "geometry"]],
                           max_distance=35, distance_col="d")
    E["fixture"] = E["seg_id"].map(fx.groupby("seg_id")["Fixture_Type"].first()).fillna("none")

    # lighting bins
    v = E["light"].to_numpy()
    qs = np.quantile(v[v > 0], [0.33, 0.66]) if (v > 0).any() else [0.0, 0.0]

    def level(x):
        if x <= 0:
            return "dark"
        if x < qs[0]:
            return "dim"
        if x < qs[1]:
            return "adequate"
        return "bright"
    E["light_level"] = E["light"].apply(level)

    # spatial cell (metric) for spread
    cen = Em.geometry.centroid
    xmin, ymin, xmax, ymax = Em.total_bounds
    cx = np.clip(((cen.x - xmin) / ((xmax - xmin) / 6)).astype(int), 0, 5)
    cy = np.clip(((cen.y - ymin) / ((ymax - ymin) / 6)).astype(int), 0, 5)
    E["cell"] = cx.astype(str) + "," + cy.astype(str)

    # select ~10 per level, spread across cells, prefer imagery, keep some without
    picks, per = [], max(1, N_TARGET // 4)
    for lev in ["dark", "dim", "adequate", "bright"]:
        pool = E[E["light_level"] == lev].copy()
        pool["has_img"] = pool["n_img"] >= 3
        pool = pool.sort_values(["has_img", "n_img"], ascending=[False, False])
        chosen, used = [], set()
        for _, r in pool.iterrows():
            if len(chosen) >= per:
                break
            if r["cell"] not in used:
                chosen.append(r["seg_id"]); used.add(r["cell"])
        for _, r in pool.iterrows():
            if len(chosen) >= per:
                break
            if r["seg_id"] not in chosen:
                chosen.append(r["seg_id"])
        picks += chosen

    sel = E[E["seg_id"].isin(picks)].copy().reset_index(drop=True)
    selm = Em[Em["seg_id"].isin(picks)].reset_index(drop=True)
    mid_m = gpd.GeoSeries(selm.geometry.interpolate(0.5, normalized=True).values, crs=METRIC)
    mid_ll = mid_m.to_crs("EPSG:4326")
    sel["mx"], sel["my"] = mid_m.x.values, mid_m.y.values
    sel["lat"], sel["lon"] = mid_ll.y.values, mid_ll.x.values

    # greedy nearest-neighbour walking order, starting from the west
    xs, ys = sel["mx"].to_numpy(), sel["my"].to_numpy()
    order, remaining = [], list(range(len(sel)))
    cur = int(np.argmin(xs))
    while remaining:
        order.append(cur); remaining.remove(cur)
        if not remaining:
            break
        dd = [(xs[j] - xs[cur])**2 + (ys[j] - ys[cur])**2 for j in remaining]
        cur = remaining[int(np.argmin(dd))]
    sel = sel.iloc[order].reset_index(drop=True)
    sel["map_no"] = np.arange(1, len(sel) + 1)

    # field sheet (predicted light deliberately NOT shown, to avoid biasing the rating)
    sheet = pd.DataFrame({
        "map_no": sel["map_no"], "segment_id": sel["seg_id"], "street": sel["street"],
        "center_lat": sel["lat"].round(6), "center_lon": sel["lon"].round(6),
        "n_images": sel["n_img"], "fixture_type": sel["fixture"],
        "datetime": "", "weather": "", "temp_f": "", "observer": "", "instrument": "",
        "lamp_context": "", "ordinal_1to4": "", "lux": "", "comfort_1to5": "",
        "photo1": "", "photo2": "", "notes": "",
    })
    sheet.to_csv(OUT / "pilot_segments.csv", index=False)

    rat = sel[["map_no", "seg_id", "street", "light", "light_level", "n_img", "fixture", "cell"]]
    rat.columns = ["map_no", "segment_id", "street", "pred_light", "pred_level",
                   "n_images", "fixture", "cell"]
    (OUT / "pilot_selection.json").write_text(rat.to_json(orient="records", indent=2))

    fig, ax = plt.subplots(figsize=(11, 11))
    Em.plot(ax=ax, color="0.85", linewidth=0.6)
    selm.plot(ax=ax, color="crimson", linewidth=2.6)
    for _, r in sel.iterrows():
        ax.annotate(str(int(r["map_no"])), (r["mx"], r["my"]), fontsize=8, weight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="crimson", lw=0.8))
    ax.set_title(f"Waylit night-pilot route: {len(sel)} segments (downtown Boise)")
    ax.set_axis_off()
    fig.savefig(OUT / "pilot_map.png", dpi=140, bbox_inches="tight")

    print(json.dumps({
        "clean_segment_coverage_pct": cov,
        "selected": int(len(sel)),
        "by_level": sel["light_level"].value_counts().to_dict(),
        "with_imagery_3plus": int((sel["n_img"] >= 3).sum()),
        "without_imagery": int((sel["n_img"] == 0).sum()),
        "distinct_cells": int(sel["cell"].nunique()),
    }, indent=2))


if __name__ == "__main__":
    main()

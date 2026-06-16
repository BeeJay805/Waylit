"""Build the downtown walk graph + System A structured lighting estimate.

Outputs to data/processed/:
  system_a_edges.gpkg     walk edges with a 'light' score
  system_a_lighting.png   a quick map for eyeballing
  system_a_stats.json     summary numbers

Run:  python scripts/build_system_a.py
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
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def streetlights_gdf():
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    bbox = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    feats, off = [], 0
    while True:
        page = arcgis.query(
            sl, where="Retired_Date IS NULL", outFields="Wattage,Height",
            geometry=bbox, geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outSR="4326",
            returnGeometry="true", resultOffset=off, resultRecordCount=2000,
        ).get("features", [])
        feats += page
        if len(page) < 2000:
            break
        off += 2000
    rows = [{"Wattage": f["attributes"].get("Wattage"),
             "Height": f["attributes"].get("Height"),
             "x": f["geometry"]["x"], "y": f["geometry"]["y"]}
            for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(
        pd.DataFrame(rows),
        geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]),
        crs="EPSG:4326").to_crs(METRIC)
    g["Height"] = g["Height"].astype(float) * 0.3048  # feet -> metres
    return g


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    print("downloading walk graph ...")
    G = ox.graph_from_polygon(poly, network_type="walk", retain_all=True)
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(METRIC).reset_index(drop=True)
    print(f"  edges: {len(edges)}")

    print("loading streetlights ...")
    lamps = streetlights_gdf()
    print(f"  active lamps in bbox: {len(lamps)}")

    print("computing lighting scores ...")
    edges["light"] = lighting.lighting_scores(edges, lamps)
    v = edges["light"].to_numpy()
    p95 = float(np.percentile(v[v > 0], 95)) if (v > 0).any() else 1.0
    edges["light_norm"] = np.clip(v / p95, 0, 1)

    out = edges[["light", "light_norm", "geometry"]].copy()
    try:
        out.to_file(OUT / "system_a_edges.gpkg", driver="GPKG")
    except Exception as e:
        print("  (gpkg write skipped:", e, ")")

    fig, ax = plt.subplots(figsize=(11, 11))
    out.plot(ax=ax, column="light_norm", cmap="magma", linewidth=1.3, legend=True)
    lamps.plot(ax=ax, color="cyan", markersize=1.5, alpha=0.35)
    ax.set_title("Waylit System A: structured lighting estimate (downtown Boise)")
    ax.set_axis_off()
    fig.savefig(OUT / "system_a_lighting.png", dpi=130, bbox_inches="tight")

    stats = {"edges": len(edges), "lamps": len(lamps),
             "light_median": float(np.median(v)), "light_max": float(v.max()),
             "edges_zero_light": int((v == 0).sum()),
             "pct_edges_dark": round(100 * (v == 0).mean(), 1)}
    (OUT / "system_a_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("saved map:", OUT / "system_a_lighting.png")


if __name__ == "__main__":
    main()

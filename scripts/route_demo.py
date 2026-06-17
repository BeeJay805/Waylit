"""Waylit routing demo (System A, lighting-based comfort).

Compares the fastest walking route with a 'calmer' route that prefers better-lit streets,
and explains the trade-off. Comfort here is the structured lighting estimate only; it will
be enriched with more features later. No night labels needed.

  edge cost = walk_time * (1 + LAMBDA * (1 - comfort))   # comfort in [0,1]

Outputs to data/processed/:
  route_demo.png   the two routes on the street map
  route_demo.json  distances, times, and mean lighting for each route

Run:  python scripts/route_demo.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
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
WALK_SPEED = 1.4   # m/s
LAMBDA = 5.0       # how strongly the calmer route avoids dark streets
A = (43.6055, -116.2120)   # (lat, lon) start
Z = (43.6240, -116.1965)   # (lat, lon) end
OUT = ROOT / "data" / "processed"


def lamps_gdf():
    import geopandas as gpd
    import pandas as pd
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    bbox = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
    feats, off = [], 0
    while True:
        page = arcgis.query(
            sl, where="Retired_Date IS NULL", outFields="Wattage,Height",
            geometry=bbox, geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outSR="4326", returnGeometry="true",
            resultOffset=off, resultRecordCount=2000).get("features", [])
        feats += page
        if len(page) < 2000:
            break
        off += 2000
    rows = [{"Wattage": f["attributes"].get("Wattage"), "Height": f["attributes"].get("Height"),
             "x": f["geometry"]["x"], "y": f["geometry"]["y"]} for f in feats if f.get("geometry")]
    g = gpd.GeoDataFrame(pd.DataFrame(rows),
                         geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]),
                         crs="EPSG:4326").to_crs(METRIC)
    g["Height"] = g["Height"].astype(float) * 0.3048
    return g


def route_stats(G, route):
    edges = ox.routing.route_to_gdf(G, route, weight="length")
    dist = float(edges["length"].sum())
    comfort = float(np.average(edges["comfort"], weights=edges["length"]))
    return dist, dist / WALK_SPEED / 60.0, comfort, edges


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.graph_from_polygon(poly, network_type="walk", retain_all=False)
    edges = ox.graph_to_gdfs(G, nodes=False)
    em = edges.to_crs(METRIC)
    light = np.array(lighting.lighting_scores(em, lamps_gdf()))
    p95 = np.percentile(light[light > 0], 95) if (light > 0).any() else 1.0
    comfort = dict(zip(edges.index, np.clip(light / p95, 0, 1)))

    for u, v, k, data in G.edges(keys=True, data=True):
        c = float(comfort.get((u, v, k), 0.0))
        t = data.get("length", 1.0) / WALK_SPEED
        data["comfort"] = c
        data["cost_fast"] = t
        data["cost_calm"] = t * (1 + LAMBDA * (1 - c))

    o = ox.distance.nearest_nodes(G, X=A[1], Y=A[0])
    d = ox.distance.nearest_nodes(G, X=Z[1], Y=Z[0])
    r_fast = nx.shortest_path(G, o, d, weight="cost_fast")
    r_calm = nx.shortest_path(G, o, d, weight="cost_calm")

    df, tf, cf, ef = route_stats(G, r_fast)
    dc, tc, cc, ec = route_stats(G, r_calm)
    result = {
        "fastest": {"dist_m": round(df), "min": round(tf, 1), "mean_lighting": round(cf, 3)},
        "calmer": {"dist_m": round(dc), "min": round(tc, 1), "mean_lighting": round(cc, 3)},
        "extra_distance_pct": round(100 * (dc - df) / df, 1) if df else 0,
        "extra_minutes": round(tc - tf, 1),
        "lighting_gain": round(cc - cf, 3),
        "identical": r_fast == r_calm,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "route_demo.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(11, 11))
    em.plot(ax=ax, color="0.88", linewidth=0.6)
    ef.to_crs(METRIC).plot(ax=ax, color="#1f77b4", linewidth=3, label="fastest")
    ec.to_crs(METRIC).plot(ax=ax, color="#ff7f0e", linewidth=3, label="calmer (better lit)",
                           linestyle=(0, (4, 2)))
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy([A[1], Z[1]], [A[0], Z[0]]), crs="EPSG:4326").to_crs(METRIC)
    ax.scatter(pts.x, pts.y, c="black", s=60, zorder=5)
    ax.annotate("A", (pts.x.iloc[0], pts.y.iloc[0]), fontsize=12, weight="bold")
    ax.annotate("B", (pts.x.iloc[1], pts.y.iloc[1]), fontsize=12, weight="bold")
    ax.legend(loc="upper right")
    ax.set_title("Waylit route demo: fastest vs calmer (better-lit) walk")
    ax.set_axis_off()
    fig.savefig(OUT / "route_demo.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "route_demo.png")


if __name__ == "__main__":
    main()

"""Waylit routing demo v2: multi-factor comfort (lighting + business activity).

Comfort combines normalized lighting and nearby business density (eyes on the street).
WEIGHTS ARE PROVISIONAL and clearly labeled; the real weights will be calibrated from the
night pilot's pairwise judgments, not hand-set. Enclosure and visual features are computed
in the feature table but kept OUT of this score because their direction is not yet known.

  comfort   = 0.6*light_norm + 0.4*business_norm     (PROVISIONAL)
  edge cost = walk_time * (1 + LAMBDA*(1 - comfort))

Outputs: data/processed/route_demo_v2.png + .json
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import yaml  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import box  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from waylit import arcgis, lighting  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
ORG = CFG["arcgis_org_base"]
METRIC = "EPSG:32611"
BBOX = f'{B["min_lon"]},{B["min_lat"]},{B["max_lon"]},{B["max_lat"]}'
ENV = "esriGeometryEnvelope"
WALK_SPEED = 1.4
LAMBDA = 5.0
A = (43.6055, -116.2120)
Z = (43.6240, -116.1965)
OUT = ROOT / "data" / "processed"


def lamps():
    sl = f"{ORG}/Boise_Streetlights_Open_Data/FeatureServer/0"
    import pandas as pd
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


def stats(G, route):
    e = ox.routing.route_to_gdf(G, route, weight="length")
    dist = float(e["length"].sum())
    return (dist, dist / WALK_SPEED / 60.0,
            float(np.average(e["comfort"], weights=e["length"])),
            float(np.average(e["lightn"], weights=e["length"])),
            float(np.average(e["poin"], weights=e["length"])), e)


def main():
    poly = box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
    G = ox.graph_from_polygon(poly, network_type="walk", retain_all=False)
    edges = ox.graph_to_gdfs(G, nodes=False)
    em = edges.to_crs(METRIC)

    light = np.array(lighting.lighting_scores(em, lamps()))
    lp95 = np.percentile(light[light > 0], 95) if (light > 0).any() else 1.0
    lightn = np.clip(light / lp95, 0, 1)

    places = gpd.read_parquet(ROOT / "data/raw/overture/places.parquet").to_crs(METRIC)
    ptree = STRtree(places.geometry.values)
    poi = np.array([len(ptree.query(g.buffer(50))) for g in em.geometry], dtype=float)
    pp90 = np.percentile(poi[poi > 0], 90) if (poi > 0).any() else 1.0
    poin = np.clip(poi / pp90, 0, 1)

    comfort = 0.6 * lightn + 0.4 * poin
    cmap = dict(zip(edges.index, comfort))
    lmap = dict(zip(edges.index, lightn))
    pmap = dict(zip(edges.index, poin))
    for u, v, k, data in G.edges(keys=True, data=True):
        c = float(cmap.get((u, v, k), 0.0))
        t = data.get("length", 1.0) / WALK_SPEED
        data["comfort"], data["lightn"], data["poin"] = c, float(lmap.get((u, v, k), 0.0)), float(pmap.get((u, v, k), 0.0))
        data["cost_fast"] = t
        data["cost_calm"] = t * (1 + LAMBDA * (1 - c))

    o = ox.distance.nearest_nodes(G, X=A[1], Y=A[0])
    d = ox.distance.nearest_nodes(G, X=Z[1], Y=Z[0])
    rf = nx.shortest_path(G, o, d, weight="cost_fast")
    rc = nx.shortest_path(G, o, d, weight="cost_calm")
    df_, tf, cf, lf, pf, ef = stats(G, rf)
    dc, tc, cc, lc, pc, ec = stats(G, rc)
    res = {"weights": "PROVISIONAL 0.6*light + 0.4*business (calibrate from pilot)",
           "fastest": {"m": round(df_), "min": round(tf, 1), "comfort": round(cf, 2),
                       "light": round(lf, 2), "business": round(pf, 2)},
           "calmer": {"m": round(dc), "min": round(tc, 1), "comfort": round(cc, 2),
                      "light": round(lc, 2), "business": round(pc, 2)},
           "extra_min": round(tc - tf, 1), "extra_pct": round(100 * (dc - df_) / df_, 1),
           "comfort_gain": round(cc - cf, 2), "identical": rf == rc}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "route_demo_v2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(11, 11))
    em.plot(ax=ax, color="0.88", linewidth=0.6)
    ef.to_crs(METRIC).plot(ax=ax, color="#1f77b4", linewidth=3, label="fastest")
    ec.to_crs(METRIC).plot(ax=ax, color="#ff7f0e", linewidth=3, linestyle=(0, (4, 2)),
                           label="calmer (lit + active)")
    pts = gpd.GeoSeries(gpd.points_from_xy([A[1], Z[1]], [A[0], Z[0]]), crs="EPSG:4326").to_crs(METRIC)
    ax.scatter(pts.x, pts.y, c="black", s=60, zorder=5)
    ax.legend(loc="upper right")
    ax.set_title("Waylit route demo v2: fastest vs calmer (lighting + business activity)")
    ax.set_axis_off()
    fig.savefig(OUT / "route_demo_v2.png", dpi=140, bbox_inches="tight")
    print("saved route_demo_v2.png")


if __name__ == "__main__":
    main()

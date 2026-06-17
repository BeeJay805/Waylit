"""Calibrated comfort router: fastest vs calmer using the LEARNED weights (not hand-picked).

Replaces the provisional 0.6*light + 0.4*business (route_demo_v2) with the weights calibrated
from the blind pairwise round (data/processed/comfort_weights.json, A_structured). Comfort per
segment = standardized structured features dotted with the learned coefficients; edge cost trades
walk time against comfort:  cost = time * (1 + LAMBDA*(1 - comfort)).

v1 BOOTSTRAP: single rater, comfort (not danger), crime never an input, weights learned not set.
Outputs: data/processed/route_calibrated.png + .json
"""
import json
import pathlib

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
B = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
METRIC = "EPSG:32611"
WALK_SPEED = 1.4          # m/s
LAMBDA = 5.0              # comfort vs time trade-off (same as the provisional demo)
A = (43.6055, -116.2120)  # origin (lat, lon)
Z = (43.6240, -116.1965)  # destination
LOG1P = {"encl_height", "poi_density", "poi_night_density"}


def seg_comfort():
    """Calibrated comfort (0..1) per feature segment + its projected midpoint, for nearest-join."""
    w = json.loads((PROC / "comfort_weights.json").read_text())["weights"]["A_structured"]
    coef = {f: w[f]["coef"] for f in w}
    df = pd.read_parquet(PROC / "segment_features.parquet")
    raw = np.zeros(len(df))
    for f, c in coef.items():
        x = df[f].astype(float).to_numpy()
        if f in LOG1P:
            x = np.log1p(np.clip(x, 0, None))
        raw += c * (x - x.mean()) / (x.std() + 1e-9)
    p5, p95 = np.percentile(raw, [5, 95])
    c01 = np.clip((raw - p5) / (p95 - p5 + 1e-9), 0, 1)
    pts = gpd.GeoSeries(gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(METRIC)
    return np.c_[pts.x.to_numpy(), pts.y.to_numpy()], c01, coef


def stats(G, route):
    e = ox.routing.route_to_gdf(G, route, weight="length")
    dist = float(e["length"].sum())
    return dist, dist / WALK_SPEED / 60.0, float(np.average(e["comfort"], weights=e["length"])), e


def main():
    sxy, c01, coef = seg_comfort()
    tree = cKDTree(sxy)
    G = ox.graph_from_polygon(box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"]),
                              network_type="walk", retain_all=False)
    E = ox.graph_to_gdfs(G, nodes=False).to_crs(METRIC)
    cent = E.geometry.centroid
    ndist, idx = tree.query(np.c_[cent.x.to_numpy(), cent.y.to_numpy()])
    cmap = dict(zip(E.index, c01[idx]))
    med = float(np.median(c01))
    for u, v, k, data in G.edges(keys=True, data=True):
        c = float(cmap.get((u, v, k), med))
        t = data.get("length", 1.0) / WALK_SPEED
        data["comfort"] = c
        data["cost_fast"] = t
        data["cost_calm"] = t * (1 + LAMBDA * (1 - c))
    print(f"learned weights: {{{', '.join(f'{f}:{c:+.2f}' for f, c in coef.items())}}}")
    print(f"comfort assigned to all {G.number_of_edges()} edges by nearest calibrated segment "
          f"(median nearest dist {np.median(ndist):.0f} m)")

    o = ox.distance.nearest_nodes(G, X=A[1], Y=A[0])
    d = ox.distance.nearest_nodes(G, X=Z[1], Y=Z[0])
    rf = nx.shortest_path(G, o, d, weight="cost_fast")
    rc = nx.shortest_path(G, o, d, weight="cost_calm")
    df_, tf, cf, ef = stats(G, rf)
    dc, tc, cc, ec = stats(G, rc)
    res = {"weights_source": "comfort_weights.json A_structured (learned, single rater)",
           "learned_coef": {f: round(c, 3) for f, c in coef.items()},
           "lambda": LAMBDA, "comfort_assignment": f"nearest segment (median {np.median(ndist):.0f} m)",
           "fastest": {"m": round(df_), "min": round(tf, 1), "comfort": round(cf, 2)},
           "calmer": {"m": round(dc), "min": round(tc, 1), "comfort": round(cc, 2)},
           "extra_min": round(tc - tf, 1), "extra_pct": round(100 * (dc - df_) / df_, 1),
           "comfort_gain": round(cc - cf, 2), "identical": rf == rc,
           "caveat": "v1 bootstrap: one rater, comfort not danger, crime never an input"}
    (PROC / "route_calibrated.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("fastest", "calmer", "extra_min", "comfort_gain")}, indent=2))

    em = ox.graph_to_gdfs(G, nodes=False)
    fig, ax = plt.subplots(figsize=(11, 11))
    em.plot(ax=ax, color="0.88", linewidth=0.6)
    ef.plot(ax=ax, color="#1f77b4", linewidth=3, label=f"fastest ({tf:.0f} min, comfort {cf:.2f})")
    ec.plot(ax=ax, color="#ff7f0e", linewidth=3, linestyle=(0, (4, 2)),
            label=f"calmer ({tc:.0f} min, comfort {cc:.2f})")
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy([A[1], Z[1]], [A[0], Z[0]]), crs="EPSG:4326")
    ax.scatter(pts.x, pts.y, c="black", s=60, zorder=5)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Waylit: fastest vs calmer with CALIBRATED comfort weights (v1 bootstrap)")
    ax.set_axis_off()
    fig.savefig(PROC / "route_calibrated.png", dpi=140, bbox_inches="tight")
    print("saved route_calibrated.png + route_calibrated.json")


if __name__ == "__main__":
    main()

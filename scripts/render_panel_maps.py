"""Render per-city maps of the AI panel's output: the real OSM walk network (faint gray)
with every rated street drawn at its true location, colored by the panel's win-rate
(how often it was picked as the safer street to walk at night). Small-multiples grid.

Win-rate per segment = wins / appearances over the city's consensus pairs (coarse: each
street is compared ~4 times). Coords come from the pairwise queues. NOT a safety surface;
a spatial view of where the panel judged streets safer vs less safe.

Run: python scripts/render_panel_maps.py
Output: data/processed/ai_panel_maps.png
"""
import collections
import csv
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
CFG = {
    "boise": (["ai_panel_consensus_boise.csv", "ai_panel_consensus_boise2.csv"],
              ["pairwise_queue.csv", "pairwise_queue2.csv"], "boise", "Boise"),
    "la": (["ai_panel_consensus_la.csv", "ai_panel_consensus_la2.csv"],
           ["la_pairwise_queue.csv", "la_pairwise_queue2.csv"], "la", "Los Angeles"),
    "slc": (["ai_panel_consensus_slc.csv"], ["slc_pairwise_queue.csv"], "slc", "Salt Lake City"),
    "denver": (["ai_panel_consensus_denver.csv"], ["denver_pairwise_queue.csv"], "denver", "Denver"),
    "dc": (["ai_panel_consensus_dc.csv"], ["dc_pairwise_queue.csv"], "dc", "Washington DC"),
    "minneapolis": (["ai_panel_consensus_minneapolis.csv"], ["minneapolis_pairwise_queue.csv"],
                    "minneapolis", "Minneapolis"),
}


def winrates(cons_files, queue_files):
    coord = {}
    for qf in queue_files:
        for r in csv.DictReader((GT / qf).open(encoding="utf-8")):
            coord[int(r["seg_a"])] = (float(r["lon_a"]), float(r["lat_a"]))
            coord[int(r["seg_b"])] = (float(r["lon_b"]), float(r["lat_b"]))
    win, app = collections.Counter(), collections.Counter()
    for cf in cons_files:
        for r in csv.DictReader((PROC / cf).open(encoding="utf-8")):
            w, l = r["consensus_winner_seg"], r["consensus_loser_seg"]
            if not w or not l:
                continue
            w, l = int(w), int(l)
            win[w] += 1
            app[w] += 1
            app[l] += 1
    return [(coord[s][0], coord[s][1], win[s] / n) for s, n in app.items() if s in coord]


def network(city_yaml, ax):
    import osmnx as ox
    from shapely.geometry import box
    B = yaml.safe_load((ROOT / "config" / "cities" / f"{city_yaml}.yaml").read_text())["bbox"]
    try:
        G = ox.graph_from_polygon(box(B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"]),
                                  network_type="walk", retain_all=True)
        E = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")
        E.plot(ax=ax, color="#c9ccd1", linewidth=0.4, zorder=1)
    except Exception as e:
        print(f"  network fetch failed for {city_yaml}: {e}")
    return B


def main():
    avail = [k for k, v in CFG.items() if (PROC / v[0][0]).exists()]
    n = len(avail)
    cols = 3 if n > 4 else 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.3 * rows))
    axes = axes.ravel() if n > 1 else [axes]
    norm, cmap = Normalize(0, 1), plt.cm.RdYlGn
    for ax, city in zip(axes, avail):
        cons, queues, cyaml, title = CFG[city]
        pts = winrates(cons, queues)
        B = network(cyaml, ax)
        lon = [p[0] for p in pts]
        lat = [p[1] for p in pts]
        wr = [p[2] for p in pts]
        ax.scatter(lon, lat, c=wr, cmap=cmap, norm=norm, s=30, edgecolors="#33333366",
                   linewidths=0.3, zorder=3)
        ml = math.radians((B["min_lat"] + B["max_lat"]) / 2)
        ax.set_aspect(1 / math.cos(ml))
        ax.set_xlim(B["min_lon"], B["max_lon"])
        ax.set_ylim(B["min_lat"], B["max_lat"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}  ({len(pts)} rated streets)", fontsize=12)
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Waylit AI panel: how safe each street was judged to feel to walk at night",
                 fontsize=15, y=0.995)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist() if hasattr(axes, "tolist") else axes,
                      fraction=0.025, pad=0.02)
    cb.set_label("panel win-rate   (red = judged less safe   to   green = judged safer)", fontsize=11)
    out = PROC / "ai_panel_maps.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"rendered {n} cities -> {out}")


if __name__ == "__main__":
    main()

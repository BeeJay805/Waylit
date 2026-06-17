"""System A: structured physical lighting-potential estimate per street segment.

For points sampled along a street segment, sum an illuminance proxy from nearby lamps:

    E  ~  Wattage * H / (d^2 + H^2)^1.5

which is the illuminance of a point source at mounting height H onto a horizontal
surface a horizontal distance d away (inverse-square with the cosine term). Segment
score = mean over sampled points. Fully auditable, no learning. Inputs must be in a
metric CRS; Wattage and Height (metres) come from the city inventory.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _sample_xy(geom, step=12.0):
    """Points every `step` metres along a projected LineString."""
    if geom is None or geom.is_empty:
        return []
    length = geom.length
    if length == 0:
        p = geom.interpolate(0)
        return [(p.x, p.y)]
    n = max(2, int(length // step) + 1)
    pts = []
    for i in range(n):
        p = geom.interpolate(min(i * step, length))
        pts.append((p.x, p.y))
    return pts


def lighting_scores(edges, lamps, radius=35.0):
    """Return mean illuminance-proxy per edge (same order as `edges`).

    edges, lamps: projected (metres) GeoDataFrames. lamps need Wattage, Height (metres).
    """
    lx = lamps.geometry.x.to_numpy()
    ly = lamps.geometry.y.to_numpy()
    watt = lamps["Wattage"].astype(float)
    watt = watt.fillna(watt.median()).to_numpy()
    hgt = lamps["Height"].astype(float)
    hgt = np.clip(hgt.fillna(hgt.median()).to_numpy(), 3.0, None)
    tree = cKDTree(np.c_[lx, ly])

    scores = []
    for geom in edges.geometry:
        vals = []
        for px, py in _sample_xy(geom):
            idx = tree.query_ball_point((px, py), radius)
            if not idx:
                vals.append(0.0)
                continue
            dx = lx[idx] - px
            dy = ly[idx] - py
            h = hgt[idx]
            e = watt[idx] * h / np.power(dx * dx + dy * dy + h * h, 1.5)
            vals.append(float(e.sum()))
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return scores

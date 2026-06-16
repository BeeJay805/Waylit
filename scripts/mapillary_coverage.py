"""Mapillary daytime-imagery coverage gate for the Waylit study area.

Reads the token from MAPILLARY_TOKEN or secrets/mapillary_token.txt, grids the downtown
bbox, and reports: coverage %, sampled density, capture-year spread, panorama share, and
viewing-direction spread. Downloads a few sample thumbnails for visual inspection and
writes a summary JSON.

Run:  python scripts/mapillary_coverage.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.parse
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPH = "https://graph.mapillary.com"
COLS, ROWS = 8, 8     # grid cells over the bbox
PER_CELL = 50         # images sampled per cell (presence + density bucket)


def token() -> str:
    t = os.environ.get("MAPILLARY_TOKEN")
    if t:
        return t.strip()
    f = ROOT / "secrets" / "mapillary_token.txt"
    if f.exists():
        return f.read_text().strip()
    raise SystemExit("No token: set MAPILLARY_TOKEN or secrets/mapillary_token.txt")


TOK = token()


def api(path: str, **params) -> dict:
    params["access_token"] = TOK
    url = f"{GRAPH}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    return {}


def load_bbox():
    import yaml
    b = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
    return b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]


def main() -> None:
    minlon, minlat, maxlon, maxlat = load_bbox()
    dlon = (maxlon - minlon) / COLS
    dlat = (maxlat - minlat) / ROWS

    covered = 0
    counts, dates, panos, angles, pool = [], [], [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            w = minlon + c * dlon
            s = minlat + r * dlat
            bbox = f"{w},{s},{w + dlon},{s + dlat}"
            data = api("images", bbox=bbox, limit=PER_CELL,
                       fields="id,captured_at,compass_angle,is_pano").get("data", [])
            counts.append(len(data))
            if data:
                covered += 1
                pool.append(data[0]["id"])
                for im in data:
                    if im.get("captured_at"):
                        dates.append(im["captured_at"])
                    panos.append(bool(im.get("is_pano")))
                    if im.get("compass_angle") is not None:
                        angles.append(im["compass_angle"])

    ncells = COLS * ROWS
    summary = {
        "grid": f"{COLS}x{ROWS}",
        "cells": ncells,
        "cells_covered": covered,
        "coverage_pct": round(100 * covered / ncells),
        "images_sampled": sum(counts),
        "cells_at_sample_cap": sum(1 for x in counts if x >= PER_CELL),
    }
    nz = sorted(x for x in counts if x)
    if nz:
        summary["density_min"] = nz[0]
        summary["density_median"] = nz[len(nz) // 2]
    if dates:
        yrs = Counter(time.gmtime(d / 1000).tm_year for d in dates)
        summary["capture_years"] = dict(sorted(yrs.items()))
        newest = time.gmtime(max(dates) / 1000)
        summary["newest"] = f"{newest.tm_year}-{newest.tm_mon:02d}"
    if panos:
        summary["panorama_pct"] = round(100 * sum(panos) / len(panos))
    if angles:
        summary["compass_octants_present"] = len({int((a % 360) // 45) for a in angles})

    for k, v in summary.items():
        print(f"{k}: {v}")

    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw" / "mapillary_coverage_summary.json").write_text(
        json.dumps(summary, indent=2))

    # download up to 4 sample thumbnails, spread across covered cells
    outdir = ROOT / "data" / "raw" / "mapillary_samples"
    outdir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pool) // 4)
    saved = []
    for i, iid in enumerate(pool[::step][:4]):
        meta = api(str(iid), fields="thumb_1024_url,captured_at,compass_angle")
        url = meta.get("thumb_1024_url")
        if url:
            p = outdir / f"sample_{i + 1}.jpg"
            urllib.request.urlretrieve(url, str(p))
            saved.append(str(p.relative_to(ROOT)))
    print("sample_images:", saved)


if __name__ == "__main__":
    main()

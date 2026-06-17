"""Reproduce the Phase 0 structured-data checks for Boise.

Run:
    python scripts/probe_structured_sources.py

Prints the counts, owner breakdown, and schemas recorded in
docs/phase0/structured-data-availability.md, so they can be re-verified live.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from waylit import arcgis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())
B = CFG["bbox"]
BBOX = (B["min_lon"], B["min_lat"], B["max_lon"], B["max_lat"])
ORG = CFG["arcgis_org_base"]


def layer(name: str, idx: int = 0) -> str:
    return f"{ORG}/{name}/FeatureServer/{idx}"


def main() -> None:
    sl = layer("Boise_Streetlights_Open_Data")
    print("STREETLIGHTS")
    print("  citywide :", arcgis.count(sl))
    print("  downtown :", arcgis.count(sl, bbox=BBOX))
    print("  by owner :", arcgis.group_count(sl, "Pole_Owner", bbox=BBOX))
    print("  by lamp  :", arcgis.group_count(sl, "Lamp_Type", bbox=BBOX))

    print("\nSCHEMAS (field names)")
    for name in ("BPR_Park_And_Street_Trees", "Boise_Buildings_3D"):
        names = [f["name"] for f in arcgis.fields(layer(name))]
        print(f"  {name}:\n    {names}")


if __name__ == "__main__":
    main()

"""Transparent DAY-walking safety score from objective pedestrian-traffic features.

Daytime walking safety is dominated by vehicle conflict, which is MEASURABLE, so this score is
reasoned from established pedestrian-safety relationships rather than learned from a rater's gut
(the rater confirmed day picks would just echo the night 'niceness' judgment). Directions and the
speed curve are grounded in published evidence; the constants are documented approximations,
auditable, and refinable against local pedestrian-crash data. Crime/demographics are never inputs.

  separation: car-free ways carry near-zero vehicle-conflict risk (safest to walk)
  speed:      struck-pedestrian fatality risk rises steeply with vehicle speed; modeled as a
              logistic centered ~35 mph (about where fatality risk passes ~50%) [Tefft 2011 / AAA]
  lanes:      more lanes = wider, longer exposure beside and crossing traffic
  sidewalk:   a sidewalk separates pedestrians from traffic and sharply cuts risk [FHWA]

day_safety in 0 (dangerous) .. 1 (safe). Adds `day_safety` to walk_safety_features.parquet.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CARFREE = 0.92          # car-free is very safe but not perfect (driveways, intersections)
SPEED_MID = 35.0        # mph where the logistic risk passes ~0.5
SPEED_SLOPE = 6.0       # mph; steepness of the fatality-vs-speed rise
SIDEWALK_CUT = 0.5      # a known sidewalk halves effective traffic exposure


def speed_risk(mph):
    return 1.0 / (1.0 + np.exp(-(mph - SPEED_MID) / SPEED_SLOPE))


def main():
    d = pd.read_parquet(PROC / "walk_safety_features.parquet")
    speed = d["maxspeed_mph"].to_numpy(dtype=float)
    lanes = d["lanes"].to_numpy(dtype=float)
    swk = d["sidewalk_present"].to_numpy(dtype=float)        # 1 protected, 0 exposed, 0.5 unknown

    sr = speed_risk(speed)
    lane_factor = np.clip(1 + 0.10 * (lanes - 2), 0.8, 1.5)
    sidewalk_mult = 1 - SIDEWALK_CUT * np.nan_to_num(swk, nan=0.5)
    risk = np.clip(sr * lane_factor * sidewalk_mult, 0, 1)
    day_safety = np.where(d["car_free"].to_numpy() == 1, CARFREE, 1 - risk)

    d["day_safety"] = np.round(day_safety, 3)
    d.to_parquet(PROC / "walk_safety_features.parquet")

    print(f"day_safety scored for {len(d)} segments (0 dangerous .. 1 safe)")
    print(f"  car-free ways: {CARFREE}")
    print("  by road class (mean day_safety):")
    for c in sorted(d[d.car_free == 0].traffic_class.unique()):
        sub = d[(d.car_free == 0) & (d.traffic_class == c)]
        nm = {2: "residential", 3: "tertiary", 4: "secondary", 5: "primary"}.get(int(c), f"class{int(c)}")
        print(f"    {nm:12s} (n={len(sub):4d}) speed~{sub.maxspeed_mph.median():.0f}mph "
              f"-> {sub.day_safety.mean():.2f}")
    print(f"  sidewalk known on roads: {(d[d.car_free==0].sidewalk_present!=0.5).mean():.0%} "
          "(Boise gap -> score leans on separation + speed)")
    q = d.day_safety.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
    print(f"  distribution p5/p25/p50/p75/p95: {list(q)}")
    print("saved day_safety into walk_safety_features.parquet")


if __name__ == "__main__":
    main()

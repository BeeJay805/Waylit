"""Build a BLIND panel manifest: K independent Claude raters judge every pair, with a
per-(rater,pair) randomized left/right placement so position bias washes out across the
panel (and per-rater L/R balance is auditable). No feature values are included -> the
read stays blind (photos only). Used to drive the ai-panel rating workflow and to map
each returned left/right vote back to a segment.

Output: cache/_panel_manifest.json = {"boise":[task,...], "la":[task,...]}
task = {city, rater, pair_id, pair_type, seg_left, seg_right, img_left, img_right,
        left_path, right_path}   (left_path/right_path absolute, for the Read tool)

Run: python scripts/ai_panel_manifest.py --raters 3
"""
import argparse
import csv
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITY = {
    "boise": {"queue": ROOT / "data/ground_truth/pairwise_queue.csv",
              "imgdir": ROOT / "data/raw/mapillary/img", "salt": 1000003},
    "la": {"queue": ROOT / "data/ground_truth/la_pairwise_queue.csv",
           "imgdir": ROOT / "data/raw/mapillary_la/img", "salt": 7700017},
    "boise2": {"queue": ROOT / "data/ground_truth/pairwise_queue2.csv",
               "imgdir": ROOT / "data/raw/mapillary/img", "salt": 3300031},
    "la2": {"queue": ROOT / "data/ground_truth/la_pairwise_queue2.csv",
            "imgdir": ROOT / "data/raw/mapillary_la/img", "salt": 5500053},
    "slc": {"queue": ROOT / "data/ground_truth/slc_pairwise_queue.csv",
            "imgdir": ROOT / "data/raw/mapillary_slc/img", "salt": 8800089},
    "denver": {"queue": ROOT / "data/ground_truth/denver_pairwise_queue.csv",
               "imgdir": ROOT / "data/raw/mapillary_denver/img", "salt": 6600067},
    "dc": {"queue": ROOT / "data/ground_truth/dc_pairwise_queue.csv",
           "imgdir": ROOT / "data/raw/mapillary_dc/img", "salt": 4400041},
    "minneapolis": {"queue": ROOT / "data/ground_truth/minneapolis_pairwise_queue.csv",
                    "imgdir": ROOT / "data/raw/mapillary_minneapolis/img", "salt": 2200023},
}


def placement(pair_id, rater, salt):
    """Deterministic per-(rater,pair) left/right; differs by rater so the same pair
    appears in mixed positions across the panel."""
    seed = (int(pair_id) * 2654435761) ^ (rater * 40503) ^ salt
    return random.Random(seed & 0xFFFFFFFF).random() < 0.5


def build_city(name, raters):
    cfg = CITY[name]
    with cfg["queue"].open(encoding="utf-8") as f:
        queue = list(csv.DictReader(f))
    tasks, missing = [], 0
    for r in range(raters):
        for p in queue:
            a_left = placement(p["pair_id"], r, cfg["salt"])
            seg_l, seg_r = (p["seg_a"], p["seg_b"]) if a_left else (p["seg_b"], p["seg_a"])
            img_l, img_r = (p["img_a"], p["img_b"]) if a_left else (p["img_b"], p["img_a"])
            pl, pr = cfg["imgdir"] / f"{img_l}.jpg", cfg["imgdir"] / f"{img_r}.jpg"
            if not (pl.exists() and pr.exists()):
                missing += 1
                continue
            tasks.append({"city": name, "rater": r, "pair_id": int(p["pair_id"]),
                          "pair_type": p["pair_type"], "seg_left": seg_l, "seg_right": seg_r,
                          "img_left": img_l, "img_right": img_r,
                          "left_path": str(pl), "right_path": str(pr)})
    return tasks, missing, len(queue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "cache/_panel_manifest.json"))
    args = ap.parse_args()
    out = {}
    for name in CITY:
        tasks, missing, npairs = build_city(name, args.raters)
        out[name] = tasks
        print(f"{name}: {npairs} pairs x {args.raters} raters = {len(tasks)} tasks "
              f"({missing} skipped for missing images)")
    pathlib.Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

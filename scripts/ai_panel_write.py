"""Consolidate AI-panel votes (left/right/same) into the per-city AI vote CSV, mapping
each vote back to a segment via the manifest. Idempotent (skips already-written
rater+pair). Panel votes never feed comfort_weights.json (routing stays human-only).

Votes JSON = [{"rater": <int 0..K-1>, "pair_id": <int>, "choice": "left"|"right"|"same"}]
(as written to disk by the panel workflow). Rater id stored as ai:claude-panel-r<rater+1>.

Run: python scripts/ai_panel_write.py --votes <votes.json> --city boise
"""
import argparse
import csv
import datetime as dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
GT = ROOT / "data" / "ground_truth"
MANIFEST = ROOT / "cache" / "_panel_manifest.json"
AI_FILE = {"boise": GT / "ai_pairwise.csv", "la": GT / "ai_pairwise_la.csv",
           "boise2": GT / "ai_pairwise_boise2.csv", "la2": GT / "ai_pairwise_la2.csv",
           "slc": GT / "ai_pairwise_slc.csv", "denver": GT / "ai_pairwise_denver.csv",
           "dc": GT / "ai_pairwise_dc.csv", "minneapolis": GT / "ai_pairwise_minneapolis.csv"}
FIELDS = ["ts_iso", "rater", "pair_id", "pair_type", "seg_left", "seg_right",
          "img_left", "img_right", "side_of_a", "choice", "winner_seg", "loser_seg",
          "ans_order1", "ans_order2", "consistent"]
CH = {"left": "L", "right": "R", "same": "tie"}


def done(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["rater"], int(r["pair_id"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--city", required=True, choices=list(AI_FILE))
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())[args.city]
    idx = {(t["rater"], t["pair_id"]): t for t in manifest}
    votes = json.loads(pathlib.Path(args.votes).read_text())
    path = AI_FILE[args.city]
    already = done(path)

    new = not path.exists()
    wrote = skipped = bad = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for v in votes:
            rater_i, pid = int(v["rater"]), int(v["pair_id"])
            ch = CH.get(str(v.get("choice", "")).lower().strip())
            rid = f"ai:claude-panel-r{rater_i + 1}"
            t = idx.get((rater_i, pid))
            if ch is None or t is None:
                bad += 1
                continue
            if (rid, pid) in already:
                skipped += 1
                continue
            winner = t["seg_left"] if ch == "L" else t["seg_right"] if ch == "R" else ""
            loser = t["seg_right"] if ch == "L" else t["seg_left"] if ch == "R" else ""
            w.writerow({"ts_iso": dt.datetime.now().isoformat(timespec="seconds"), "rater": rid,
                        "pair_id": pid, "pair_type": t["pair_type"],
                        "seg_left": t["seg_left"], "seg_right": t["seg_right"],
                        "img_left": t["img_left"], "img_right": t["img_right"],
                        "side_of_a": "", "choice": ch, "winner_seg": winner, "loser_seg": loser,
                        "ans_order1": "panel", "ans_order2": "", "consistent": 1})
            already.add((rid, pid))
            wrote += 1
    print(f"{args.city}: wrote {wrote}, skipped {skipped} (already), bad {bad} -> {path.name}")


if __name__ == "__main__":
    main()

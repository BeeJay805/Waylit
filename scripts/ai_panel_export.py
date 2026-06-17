"""Export the AI-panel consensus dataset for a city: one row per pair with the consensus
winner/loser segment, the number of raters, the agreement fraction, a confidence tier, and
whether it matches the human rater. This is the labeled 'DNA' (pairwise night-safety
preferences from a blind, debiased, multi-read Claude panel). It is a research label set,
NOT a routing input - comfort_weights.json stays human-only.

Run: python scripts/ai_panel_export.py --city boise
Output: data/processed/ai_panel_consensus_<city>.csv
"""
import argparse
import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ai_panel_analyze as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(A.CITY))
    args = ap.parse_args()
    cfg = A.CITY[args.city]

    hw, _ = A.read_decisive(cfg["human"])
    aw, _ = A.read_decisive(cfg["ai"])
    human = next((r for r in hw if not r.startswith("ai:")), None)
    panel = sorted(r for r in aw if r.startswith(A.PANEL_PREFIX))

    # recover loser per (pair, winner) from the ai file
    loser_of = collections.defaultdict(dict)
    with cfg["ai"].open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["rater"] in panel and row.get("choice") in ("L", "R"):
                loser_of[int(row["pair_id"])][str(row["winner_seg"])] = str(row["loser_seg"])

    pairs = sorted(set().union(*[set(aw[r]) for r in panel]))
    out = A.PROC / f"ai_panel_consensus_{args.city}.csv"
    tiers = collections.Counter()
    rows = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "consensus_winner_seg", "consensus_loser_seg", "n_raters",
                    "top_votes", "agreement_frac", "confidence_tier", "human_winner_seg",
                    "matches_human"])
        for pid in pairs:
            segs = [aw[r][pid] for r in panel if pid in aw.get(r, {})]
            if len(segs) < 2:
                continue
            c = collections.Counter(segs)
            win, n = c.most_common(1)[0]
            frac = n / len(segs)
            tier = ("unanimous" if frac == 1 else "strong" if frac >= 0.7 else "split")
            tiers[tier] += 1
            lose = loser_of.get(pid, {}).get(win, "")
            hwin = hw.get(human, {}).get(pid, "")
            match = "" if not hwin else ("yes" if hwin == win else "no")
            w.writerow([pid, win, lose, len(segs), n, round(frac, 3), tier, hwin, match])
            rows += 1
    print(f"[{args.city}] wrote {rows} consensus pairs -> {out.name}  tiers={dict(tiers)}")


if __name__ == "__main__":
    main()

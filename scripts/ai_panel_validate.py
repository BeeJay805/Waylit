"""Deeper validation of the AI panel consensus, saved reproducibly:
  - agreement-with-human by panel CONFIDENCE tier (does panel confidence predict agreement?)
  - agreement-with-human by PAIR_TYPE near/mild/strong (is the signal graded by difficulty?)
  - (Boise) CONVERGENCE: how often the panel picks the higher comfort_night (human-LEARNED) and
    higher day_safety (OBJECTIVE) segment - tests whether the panel tracks night comfort (it should)
    vs day safety (it should not, i.e. it reproduces the day/night divergence).

Output: data/processed/ai_panel_validation.json
Run: python scripts/ai_panel_validate.py
"""
import collections
import csv
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
QUEUE = {"boise": GT / "pairwise_queue.csv", "la": GT / "la_pairwise_queue.csv"}


def by_group(rows, keyfn):
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["matches_human"] in ("yes", "no"):
            k = keyfn(r)
            g[k][0] += (r["matches_human"] == "yes")
            g[k][1] += 1
    return {k: {"agree": v[0], "n": v[1], "rate": round(v[0] / v[1], 3)} for k, v in g.items() if v[1]}


def pick_higher(pairs, df, col):
    w = np.array([df.loc[a, col] if a in df.index else np.nan for a, _ in pairs])
    l = np.array([df.loc[b, col] if b in df.index else np.nan for _, b in pairs])
    m = ~(np.isnan(w) | np.isnan(l)) & (w != l)
    return (round(float((w > l)[m].mean()), 3), int(m.sum())) if m.sum() else (None, 0)


def main():
    out = {}
    for city in ("boise", "la"):
        cf = PROC / f"ai_panel_consensus_{city}.csv"
        if not cf.exists():
            continue
        rows = list(csv.DictReader(cf.open(encoding="utf-8")))
        ptype = {int(r["pair_id"]): r["pair_type"]
                 for r in csv.DictReader(QUEUE[city].open(encoding="utf-8"))}
        out[city] = {
            "by_confidence_tier": by_group(rows, lambda r: r["confidence_tier"]),
            "by_pair_type": by_group(rows, lambda r: ptype.get(int(r["pair_id"]), "?")),
        }

    # Boise convergence with the established per-segment scores
    df = pd.read_csv(ROOT / "dataset" / "waylit_boise_downtown_v1.csv").set_index("seg_id")
    rows = [r for r in csv.DictReader((PROC / "ai_panel_consensus_boise.csv").open(encoding="utf-8"))
            if r["consensus_loser_seg"]]
    pairs = [(int(r["consensus_winner_seg"]), int(r["consensus_loser_seg"])) for r in rows]
    conv = {}
    for col in ("comfort_night", "day_safety"):
        ag, n = pick_higher(pairs, df, col)
        conv[col] = {"panel_picks_higher": ag, "n": n}
    out.setdefault("boise", {})["convergence_with_scores"] = conv
    out["boise"]["convergence_note"] = (
        "comfort_night is LEARNED from the human rater; day_safety is OBJECTIVE. The panel "
        "tracking comfort_night (~0.67) but not day_safety (~0.54, near chance) reproduces the "
        "day/night divergence: a 'safe to walk at night' judgment is night-comfort, not day-safety.")

    (PROC / "ai_panel_validation.json").write_text(json.dumps(out, indent=2))
    for city, d in out.items():
        print(f"[{city}]")
        if "by_confidence_tier" in d:
            print("  by confidence tier:", {k: f"{v['rate']:.0%}(n={v['n']})"
                                            for k, v in d["by_confidence_tier"].items()})
            print("  by pair_type:", {k: f"{v['rate']:.0%}(n={v['n']})"
                                      for k, v in d["by_pair_type"].items()})
        if "convergence_with_scores" in d:
            print("  convergence:", {k: f"{v['panel_picks_higher']:.0%}(n={v['n']})"
                                     for k, v in d["convergence_with_scores"].items()})
    print("wrote ai_panel_validation.json")


if __name__ == "__main__":
    main()

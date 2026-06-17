"""Compare AI pseudo-rater votes against the human rater and against each other,
reusing the canonical fit functions. Reports:
  - per-rater decisive / tie / skip counts
  - agreement matrix: every rater-pair's agreement on shared decisive pairs, with
    kappa-vs-chance ((p-0.5)/0.5), so human-vs-model and model-vs-model are comparable
  - single-feature drivers: the parameter-free 'pick the segment with more of feature X'
    agreement for each rater - does a model key on the same features (light, night
    business, transit) the human did?

Reuses fit_pairwise_models.single_feature_agreement and its STRUCT feature set, so this
is a diagnostic over the SAME canonical features, not a second scoring system. It does
NOT fit or modify comfort_weights.json - the human-learned routing weights stay
human-only. A VLM is not the target demographic, so none of this validates that the
human night-safety tilt generalizes across people; it measures machine reproducibility.

Run after rating:  python scripts/ai_vs_human.py
"""
import collections
import csv
import itertools
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_pairwise_models as F  # noqa: E402

GT = F.ROOT / "data" / "ground_truth"
HUMAN = GT / "pairwise.csv"
AI = GT / "ai_pairwise.csv"
OUT = F.PROC / "ai_vs_human.json"
MIN_SHARED = 10   # below this, an agreement number is too noisy to read


def read_rows(path):
    """Yield (rater, pair_id, choice, winner_seg, loser_seg) for every logged row."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield (r.get("rater", "?"), int(r["pair_id"]), r.get("choice", ""),
                   r.get("winner_seg"), r.get("loser_seg"))


def main():
    rows = list(read_rows(HUMAN)) + list(read_rows(AI))
    if not rows:
        print("no votes found in", HUMAN, "or", AI)
        return

    counts = collections.defaultdict(lambda: collections.Counter())
    decisive = collections.defaultdict(list)          # rater -> [(winner, loser)]
    by_pair = collections.defaultdict(dict)            # pair_id -> {rater: winner_seg}
    for rater, pid, choice, w, l in rows:
        counts[rater][choice if choice in ("L", "R", "tie", "skip") else "skip"] += 1
        if choice in ("L", "R") and w and l:
            decisive[rater].append((int(w), int(l)))
            by_pair[pid][rater] = int(w)

    raters = sorted(counts, key=lambda r: (r.startswith("ai:"), r))   # human first

    # agreement matrix over every rater pair, on pairs both rated decisively
    matrix = []
    for a, b in itertools.combinations(raters, 2):
        agree = tot = 0
        for v in by_pair.values():
            if a in v and b in v:
                tot += 1
                agree += int(v[a] == v[b])
        if tot:
            pa = agree / tot
            matrix.append({"a": a, "b": b, "shared_decisive": tot,
                           "agreement": round(pa, 3),
                           "kappa_vs_chance": round((pa - 0.5) / 0.5, 3),
                           "underpowered": tot < MIN_SHARED})

    # single-feature drivers per rater (reuse the canonical, leak-free function)
    raw = pd.read_parquet(F.PROC / "segment_features.parquet").set_index("seg_id")
    drivers = {}
    for r in raters:
        if len(decisive[r]) >= F.MIN_PAIRS:
            sfa = F.single_feature_agreement(decisive[r], raw)
            drivers[r] = {"n_decisive": len(decisive[r]),
                          "top": dict(list(sfa.items())[:3]),
                          "light": sfa.get("light"),
                          "poi_night_density": sfa.get("poi_night_density")}
        else:
            drivers[r] = {"n_decisive": len(decisive[r]), "note": "too few pairs to read drivers"}

    human = [r for r in raters if not r.startswith("ai:")]
    ai = [r for r in raters if r.startswith("ai:")]
    vs_human = [m for m in matrix if (m["a"] in human) != (m["b"] in human)]
    model_model = [m for m in matrix if m["a"] in ai and m["b"] in ai]

    caveats = [
        "AI pseudo-raters are NOT the target demographic (women who walk alone at night) "
        "and do NOT validate that the human night-safety tilt generalizes across people.",
        "Model-vs-model agreement measures machine-perception reproducibility / shared "
        "training priors, not human consensus.",
        "Daytime photos cannot show night illumination; a model 'safer at night' pick keys "
        "on visible street niceness, the same proxy the human reported using.",
        "These votes are kept out of comfort_weights.json; routing weights stay human-only.",
    ]
    result = {
        "raters": {r: dict(counts[r]) for r in raters},
        "agreement_matrix": matrix,
        "human_vs_model": vs_human,
        "model_vs_model": model_model,
        "single_feature_drivers": drivers,
        "human_feature_anchors": {"note": "human rater1 from fit: light 0.71, "
                                  "poi_night 0.68, transit 0.67"},
        "caveats": caveats,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("per-rater decisive/tie/skip:")
    for r in raters:
        c = counts[r]
        print(f"  {r:18s} dec={c['L']+c['R']:3d}  tie={c['tie']:3d}  skip={c['skip']:3d}")
    print("\nagreement matrix (shared decisive pairs):")
    for m in matrix:
        flag = "  [underpowered]" if m["underpowered"] else ""
        print(f"  {m['a']:18s} vs {m['b']:18s}  n={m['shared_decisive']:3d}  "
              f"agree={m['agreement']:.0%}  kappa={m['kappa_vs_chance']:+.2f}{flag}")
    print("\nsingle-feature drivers (pick higher X matches rater):")
    for r in raters:
        d = drivers[r]
        if "top" in d:
            tops = ", ".join(f"{k} {v['agreement']:.0%}" for k, v in d["top"].items())
            print(f"  {r:18s} (n={d['n_decisive']:3d})  {tops}")
        else:
            print(f"  {r:18s} (n={d['n_decisive']:3d})  {d['note']}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()

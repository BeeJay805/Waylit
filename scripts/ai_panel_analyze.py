"""Analyze the AI panel for a city: per-pair consensus (majority of the 3 raters) with
confidence, inter-rater reliability, position-bias per rater, consensus-vs-human agreement
(+ kappa), per-rater-vs-human, single-feature drivers of the consensus labels, and the list
of low-confidence (2-1 split) pairs the loop should firm up. Panel votes never feed
comfort_weights.json (routing stays human-only).

Reuses fit_pairwise_models.single_feature_agreement / STRUCT for the canonical, leak-free
driver read. Drivers are skipped if the city's feature parquet lacks those columns.

Run: python scripts/ai_panel_analyze.py --city boise
"""
import argparse
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
PROC = F.PROC
PANEL_PREFIX = "ai:claude-panel-r"   # raters detected dynamically (K=3 base, more after firm-up)
CITY = {
    "boise": {"human": GT / "pairwise.csv", "ai": GT / "ai_pairwise.csv",
              "feats": PROC / "segment_features.parquet", "ref": "ai:claude-opus-4.8"},
    "la": {"human": GT / "la_pairwise.csv", "ai": GT / "ai_pairwise_la.csv",
           "feats": PROC / "la_seg_features.parquet", "ref": None},
    "boise2": {"human": None, "ai": GT / "ai_pairwise_boise2.csv",   # batch-2: no human ratings
               "feats": PROC / "segment_features.parquet", "ref": "ai:claude-opus-4.8"},
    "la2": {"human": None, "ai": GT / "ai_pairwise_la2.csv",
            "feats": PROC / "la_seg_features.parquet", "ref": None},
    "slc": {"human": None, "ai": GT / "ai_pairwise_slc.csv",
            "feats": PROC / "slc_features.parquet", "ref": None},
    "denver": {"human": None, "ai": GT / "ai_pairwise_denver.csv",
               "feats": PROC / "denver_features.parquet", "ref": None},
    "dc": {"human": None, "ai": GT / "ai_pairwise_dc.csv",
           "feats": PROC / "dc_features.parquet", "ref": None},
    "minneapolis": {"human": None, "ai": GT / "ai_pairwise_minneapolis.csv",
                    "feats": PROC / "minneapolis_features.parquet", "ref": None},
}


def read_decisive(path):
    """rater -> {pair_id: winner_seg(str)}, plus rater -> Counter(choice) for bias."""
    wins = collections.defaultdict(dict)
    choice = collections.defaultdict(collections.Counter)
    if path is None or not path.exists():
        return wins, choice
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            choice[r["rater"]][r.get("choice", "")] += 1
            if r.get("choice") in ("L", "R") and r.get("winner_seg"):
                wins[r["rater"]][int(r["pair_id"])] = str(r["winner_seg"])
    return wins, choice


# per-city driver feature set (NO crime/income/race - those are offline audit only)
FEATS = {
    "boise": ["light", "encl_frontage", "encl_height", "poi_density", "poi_night_density",
              "transit_night_400m"],
    "la": ["lit_frac", "encl_frontage", "encl_height", "poi_density", "poi_night_density"],
    "boise2": ["light", "encl_frontage", "encl_height", "poi_density", "poi_night_density",
               "transit_night_400m"],
    "la2": ["lit_frac", "encl_frontage", "encl_height", "poi_density", "poi_night_density"],
    "slc": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"],
    "denver": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"],
    "dc": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"],
    "minneapolis": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"],
}


def single_feature(pairs, raw, feats):
    """Parameter-free: how often 'pick the segment with more of feature X' matches the label."""
    import numpy as np
    out = {}
    for c in feats:
        if c not in raw.columns:
            continue
        wv = np.array([raw.loc[w, c] for w, _ in pairs], dtype=float)
        lv = np.array([raw.loc[l, c] for _, l in pairs], dtype=float)
        m = wv != lv
        if m.sum() == 0:
            continue
        out[c] = {"agreement": round(float((wv > lv)[m].mean()), 3), "n_differ": int(m.sum())}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["agreement"]))


def kappa(p):
    return round((p - 0.5) / 0.5, 3)


def agree(a, b):
    """fraction of pairs both rated decisively where winners match, + n."""
    shared = set(a) & set(b)
    if not shared:
        return None, 0
    return round(sum(a[p] == b[p] for p in shared) / len(shared), 3), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITY))
    args = ap.parse_args()
    cfg = CITY[args.city]

    hw, _ = read_decisive(cfg["human"])
    aw, ach = read_decisive(cfg["ai"])
    human_raters = [r for r in hw if not r.startswith("ai:")]
    human = human_raters[0] if human_raters else None
    panel_raters = sorted(r for r in aw if r.startswith(PANEL_PREFIX))
    base = panel_raters[:3]   # the first 3 raters; reliability is reported on these

    # per-pair consensus over ALL present panel raters (K=3 base, K=5 after firm-up)
    pairs = set().union(*[set(aw[r]) for r in panel_raters]) if panel_raters else set()
    consensus = {}
    conf = collections.Counter()
    low_conf = []
    for pid in sorted(pairs):
        segs = [aw[r][pid] for r in panel_raters if pid in aw.get(r, {})]
        if len(segs) < 2:
            continue
        c = collections.Counter(segs)
        top, n = c.most_common(1)[0]
        consensus[pid] = top
        conf[f"{n}/{len(segs)}"] += 1
        if n / len(segs) < 0.7:               # 2/3 or 3/5 = contested -> low confidence
            low_conf.append(pid)

    # reliability on the BASE 3 raters (comparable across runs)
    unanimous = three = 0
    for pid in pairs:
        segs = [aw[r][pid] for r in base if pid in aw.get(r, {})]
        if len(segs) == 3:
            three += 1
            if len(set(segs)) == 1:
                unanimous += 1
    pair_agrees = []
    for a, b in itertools.combinations(base, 2):
        ag, n = agree(aw.get(a, {}), aw.get(b, {}))
        if ag is not None:
            pair_agrees.append(ag)
    mean_panel_agreement = round(sum(pair_agrees) / len(pair_agrees), 3) if pair_agrees else None

    # consensus vs human + per-rater vs human + ref(claude-opus) vs consensus
    cons_vs_human = agree(consensus, hw.get(human, {})) if human else (None, 0)
    per_rater_vs_human = {r: agree(aw.get(r, {}), hw.get(human, {})) for r in panel_raters} if human else {}
    ref = cfg["ref"]
    ref_vs_consensus = agree(aw.get(ref, {}), consensus) if ref and ref in aw else (None, 0)

    # position bias per rater (choice L/R = screen side picked)
    bias = {}
    for r in panel_raters:
        c = ach.get(r, {})
        dec = c.get("L", 0) + c.get("R", 0)
        bias[r] = {"L": c.get("L", 0), "R": c.get("R", 0), "tie": c.get("tie", 0),
                   "L_share": round(c.get("L", 0) / dec, 3) if dec else None}

    # drivers of the consensus labels (winner, loser) - reuse canonical function if cols exist
    drivers = {}
    bias_audit = {}
    try:
        raw = pd.read_parquet(cfg["feats"]).set_index("seg_id")
        raw.index = raw.index.astype(str)
        feats = FEATS[args.city]
        have = [c for c in feats if c in raw.columns]
        if len(have) >= 3:
            # build (winner, loser) for consensus pairs: loser = the other seg from any panel rater
            cons_pairs = []
            # recover loser from the ai file rows
            loser_of = {}
            with cfg["ai"].open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["rater"] in panel_raters and row.get("choice") in ("L", "R"):
                        loser_of.setdefault(int(row["pair_id"]), {})[str(row["winner_seg"])] = str(row["loser_seg"])
            for pid, w in consensus.items():
                lo = loser_of.get(pid, {}).get(w)
                if lo is not None and w in raw.index and lo in raw.index:
                    cons_pairs.append((int(w) if str(w).isdigit() else w, int(lo) if str(lo).isdigit() else lo))
            # single_feature_agreement expects ints in index; align types
            raw2 = raw.copy()
            try:
                raw2.index = raw2.index.astype(int)
                cp = [(int(w), int(l)) for w, l in cons_pairs]
            except ValueError:
                cp = cons_pairs
                raw2 = raw
            sfa = single_feature(cp, raw2, feats)
            drivers = {"n_consensus_pairs": len(cp), "top": dict(list(sfa.items())[:5])}
            # OFFLINE bias audit: does the blind panel track income/race/crime? ~50% = clean.
            # These columns are AUDIT ONLY and never enter the judgment (the panel saw photos only).
            ba = single_feature(cp, raw2, ["income", "pct_nonwhite", "crime"])
            if ba:
                bias_audit = {k: {"pick_higher_agreement": v["agreement"], "n": v["n_differ"]}
                              for k, v in ba.items()}
        else:
            drivers = {"note": f"feature parquet lacks driver cols (have {have}); drivers skipped"}
    except Exception as e:
        drivers = {"note": f"drivers unavailable: {e}"}

    out = {
        "city": args.city,
        "n_pairs_with_3_raters": three,
        "reliability": {"unanimous": unanimous, "unanimous_rate": round(unanimous / three, 3) if three else None,
                        "confidence_breakdown": dict(conf),
                        "mean_pairwise_panel_agreement": mean_panel_agreement},
        "position_bias_per_rater": bias,
        "consensus_vs_human": {"rater": human, "agreement": cons_vs_human[0],
                               "n": cons_vs_human[1],
                               "kappa": kappa(cons_vs_human[0]) if cons_vs_human[0] is not None else None},
        "per_rater_vs_human": {r: {"agreement": v[0], "n": v[1]} for r, v in per_rater_vs_human.items()},
        "ref_vs_consensus": {"ref": ref, "agreement": ref_vs_consensus[0], "n": ref_vs_consensus[1]},
        "consensus_drivers": drivers,
        "demographic_bias_audit": bias_audit,
        "low_confidence_pairs": low_conf,
        "caveats": [
            "Panel = 3 reads of ONE model (Claude); high agreement is self-consistency, not human "
            "inter-rater independence, and not target-demographic validation.",
            "Blind to feature values; debiased by randomized placement; never feeds routing weights.",
        ],
    }
    (PROC / f"ai_panel_{args.city}.json").write_text(json.dumps(out, indent=2))

    r = out["reliability"]
    print(f"[{args.city}] pairs with 3 raters: {three}")
    print(f"  reliability: unanimous {r['unanimous']}/{three} ({r['unanimous_rate']}), "
          f"conf {r['confidence_breakdown']}, mean pairwise {r['mean_pairwise_panel_agreement']}")
    print(f"  position bias L-share: " + ", ".join(f"{k.split('-')[-1]}={v['L_share']}" for k, v in bias.items()))
    cv = out["consensus_vs_human"]
    print(f"  consensus vs human({human}): {cv['agreement']} on n={cv['n']} (kappa {cv['kappa']})")
    if ref_vs_consensus[0] is not None:
        print(f"  {ref} vs consensus: {ref_vs_consensus[0]} on n={ref_vs_consensus[1]}")
    if "top" in drivers:
        print(f"  consensus drivers ({drivers['n_consensus_pairs']} pairs): " +
              ", ".join(f"{k} {v['agreement']:.0%}" for k, v in drivers["top"].items()))
    else:
        print(f"  drivers: {drivers.get('note')}")
    if bias_audit:
        print("  demographic bias audit (pick-higher agreement, ~50% = clean): " +
              ", ".join(f"{k} {v['pick_higher_agreement']:.0%}" for k, v in bias_audit.items()))
    print(f"  low-confidence (split) pairs: {len(low_conf)}")
    print(f"  wrote ai_panel_{args.city}.json")


if __name__ == "__main__":
    main()

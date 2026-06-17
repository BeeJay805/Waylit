"""Rate a city's blind night-safety pairs with a cheap OpenAI vision model (gpt-4o-mini),
bypassing the local subagent runtime. Each pair is asked in BOTH photo orders (order-swap
debiasing); a vote counts only if the pick is stable across the swap. Reads the per-pair
layout from cache/_panel_manifest.json (rater 0 = one entry per pair). Blind: two photos
only, no scores. Reads OPENAI_API_KEY from the environment (never stored or printed).

Writes data/processed/ai_panel_consensus_<city>.csv (standard consensus schema, so it slots
into the master + maps) and prints order-consistency (reliability proxy) + feature drivers.
Resumable: skips pairs already in the output. Rater id recorded = ai:gpt-4o-mini.

Run:  OPENAI_API_KEY=... python scripts/ai_rate_openai.py --city dc
"""
import argparse
import base64
import csv
import io
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MANIFEST = ROOT / "cache" / "_panel_manifest.json"
MODEL = "gpt-4o-mini"
QUESTION = ("You are shown two daytime street photos. The FIRST image is the LEFT street, the "
            "SECOND is the RIGHT street. Judging only from the photos, which street feels safer "
            "to walk alone at night? Read the street (lighting, openness, eyes-on-street, "
            "isolation), not the daylight. You MUST choose one; do not say they are the same. "
            "Reply with exactly one word: LEFT or RIGHT.")
WORD = re.compile(r"\b(LEFT|RIGHT|SAME)\b", re.I)
FEATS = {"dc": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"],
         "minneapolis": ["encl_frontage", "encl_height", "poi_density", "poi_night_density"]}
FIELDS = ["pair_id", "consensus_winner_seg", "consensus_loser_seg", "n_raters", "top_votes",
          "agreement_frac", "confidence_tier", "human_winner_seg", "matches_human"]


def b64(path, side=512):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = side / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask(img_first, img_second, key):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4, "messages": [{"role": "user",
            "content": [{"type": "text", "text": QUESTION},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64(img_first)}", "detail": "high"}},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64(img_second)}", "detail": "high"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))      # rate limit: back off hard
            elif attempt == 5:
                raise
            else:
                time.sleep(3 * (attempt + 1))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))


def parse(ans):
    m = WORD.search(ans or "")
    return m.group(1).upper() if m else None


def done(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {int(r["pair_id"]) for r in csv.DictReader(f)}


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not in environment")

    tasks = [t for t in json.loads(MANIFEST.read_text())[args.city] if t["rater"] == 0]
    out = PROC / f"ai_panel_consensus_{args.city}.csv"
    already = done(out)
    todo = [t for t in tasks if t["pair_id"] not in already]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.city}: {len(todo)} of {len(tasks)} pairs to rate with {MODEL} "
          f"(2 calls each, order-swapped)", flush=True)

    new = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t0 = time.time()
    consistent = 0
    for i, t in enumerate(todo, 1):
        sl, sr = str(t["seg_left"]), str(t["seg_right"])
        il = ROOT / "data" / "raw" / f"mapillary_{args.city}" / "img" / f"{t['img_left']}.jpg"
        ir = ROOT / "data" / "raw" / f"mapillary_{args.city}" / "img" / f"{t['img_right']}.jpg"
        a = parse(ask(il, ir, key))                 # order A: seg_left shown first
        b = parse(ask(ir, il, key))                 # order B: swapped
        pa = sl if a == "LEFT" else sr if a == "RIGHT" else None
        pb = sr if b == "LEFT" else sl if b == "RIGHT" else None   # first image is seg_right here
        if pa is not None and pa == pb:
            win = pa
            lose = sr if win == sl else sl
            tier, frac, top = "unanimous", 1.0, 2
            consistent += 1
        elif a == "SAME" and b == "SAME":
            win = lose = ""
            tier, frac, top = "tie", 0.5, 1
        else:
            win = lose = ""
            tier, frac, top = "split", 0.5, 1
        w.writerow({"pair_id": t["pair_id"], "consensus_winner_seg": win, "consensus_loser_seg": lose,
                    "n_raters": 2, "top_votes": top, "agreement_frac": frac,
                    "confidence_tier": tier, "human_winner_seg": "", "matches_human": ""})
        f.flush()
        time.sleep(0.3)                              # gentle pacing for rate limits
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s  consistent={consistent}", flush=True)
    f.close()
    print(f"done: {args.city} | order-consistent {consistent}/{len(todo)} "
          f"({consistent/max(1,len(todo)):.0%}) in {time.time()-t0:.0f}s -> {out.name}", flush=True)

    # feature drivers of the decisive picks (parameter-free, reuse the canonical idea)
    import numpy as np
    import pandas as pd
    raw = pd.read_parquet(PROC / f"{args.city}_features.parquet").set_index("seg_id")
    raw.index = raw.index.astype(int)
    pairs = []
    with out.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["consensus_winner_seg"] and r["consensus_loser_seg"]:
                pairs.append((int(r["consensus_winner_seg"]), int(r["consensus_loser_seg"])))
    drivers = {}
    for c in FEATS[args.city]:
        wv = np.array([raw.loc[w, c] for w, _ in pairs], float)
        lv = np.array([raw.loc[l, c] for _, l in pairs], float)
        m = wv != lv
        if m.sum():
            drivers[c] = round(float((wv > lv)[m].mean()), 3)
    print("drivers:", dict(sorted(drivers.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()

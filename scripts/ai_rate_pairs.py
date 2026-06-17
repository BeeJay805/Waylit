"""Blind AI pseudo-raters: cheap vision-language models judge the SAME pairs the
human rated, under the SAME blind question (two daytime photos only - no scores, no
feature values, no map). Votes append to data/ground_truth/ai_pairwise.csv with
rater='ai:<model>', in the human pairwise.csv schema (+ audit columns) so the existing
agreement code reads them unchanged.

POSITION-BIAS DEBIASING (important): small VLMs tend to favor the first/left image
regardless of content - the SmolVLM-500M single-call pilot picked LEFT 31/40 times under
random placement. So every pair is asked TWICE, with the two photos swapped. A vote is
counted DECISIVE only when the model picks the SAME street both times; if it picks
whichever photo is on a given side (flips with the swap) the pair is 'incons' (position
biased) and NOT counted. The consistent-fraction is itself the headline: it measures how
much real, order-stable street signal the model has versus pure layout bias.

WHAT THIS IS NOT: a substitute for human multi-rater calibration. A VLM is not the target
demographic (women who walk alone at night), and models share training priors, so
agreement among models measures machine-perception reproducibility, not human consensus.
These votes stay in a SEPARATE file and never feed comfort_weights.json - the
human-learned routing weights stay human-only.

Run:  python scripts/ai_rate_pairs.py --model smolvlm-500m --limit 40
Models: smolvlm-500m (fast/weak), smolvlm-2.2b, qwen2vl-2b (stronger, slower on CPU).
"""
import argparse
import csv
import datetime as dt
import pathlib
import random
import re
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
GT = ROOT / "data" / "ground_truth"
IMGDIR = ROOT / "data" / "raw" / "mapillary" / "img"
QUEUE = GT / "pairwise_queue.csv"
LOG = GT / "ai_pairwise.csv"
LOG_FIELDS = ["ts_iso", "rater", "pair_id", "pair_type", "seg_left", "seg_right",
              "img_left", "img_right", "side_of_a", "choice", "winner_seg", "loser_seg",
              "ans_order1", "ans_order2", "consistent"]

QUESTION = ("You are shown two daytime street photos. The FIRST image is LEFT, the "
            "SECOND is RIGHT. Daytime photos stand in for a night question: read the "
            "street, not the daylight. Which street feels safer to walk alone at night? "
            "Reply with exactly one word: LEFT, RIGHT, or SAME.")

REGISTRY = {
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-2.2b": "HuggingFaceTB/SmolVLM-Instruct",
    "qwen2vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
}
WORD = re.compile(r"\b(LEFT|RIGHT|SAME|NEITHER|BOTH)\b", re.I)


def seeded_a_left(pair_id):
    """Exactly blind_compare.py's per-pair left/right shuffle."""
    return random.Random(int(pair_id) * 2654435761 & 0xFFFFFFFF).random() < 0.5


def layout(pair):
    """Canonical (human-protocol) placement: (seg_l, seg_r, img_l, img_r, side_of_a)."""
    if seeded_a_left(pair["pair_id"]):
        return (pair["seg_a"], pair["seg_b"], pair["img_a"], pair["img_b"], "L")
    return (pair["seg_b"], pair["seg_a"], pair["img_b"], pair["img_a"], "R")


def done(rater):
    if not LOG.exists():
        return set()
    with LOG.open(encoding="utf-8") as f:
        return {int(r["pair_id"]) for r in csv.DictReader(f) if r.get("rater") == rater}


def append(row):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def parse(ans):
    """First LEFT/RIGHT/SAME token -> 'L' (first image), 'R' (second), 'tie', or 'skip'."""
    m = WORD.search(ans or "")
    if not m:
        return "skip"
    return {"LEFT": "L", "RIGHT": "R", "SAME": "tie", "BOTH": "tie", "NEITHER": "tie"}[m.group(1).upper()]


def picked(choice, first_seg, second_seg):
    """Which segment the model chose, given the on-screen first/second segments."""
    return first_seg if choice == "L" else second_seg if choice == "R" else None


def load_model(model_id):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return proc, model


def cap_img(p, side=512):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = side / max(w, h)
    return im.resize((int(w * s), int(h * s))) if s < 1 else im


def judge_once(proc, model, img_first, img_second):
    """One model call: returns raw text for photos shown (first=LEFT, second=RIGHT)."""
    import torch
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"}, {"type": "text", "text": QUESTION}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    imgs = [cap_img(IMGDIR / f"{img_first}.jpg"), cap_img(IMGDIR / f"{img_second}.jpg")]
    inp = proc(text=[prompt], images=imgs, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--limit", type=int, default=0, help="rate only the first N pairs (0 = all)")
    args = ap.parse_args()

    rater = f"ai:{args.model}"
    model_id = REGISTRY[args.model]
    with QUEUE.open(encoding="utf-8") as f:
        queue = list(csv.DictReader(f))
    for r in queue:
        r["pair_id"] = int(r["pair_id"])
    if args.limit:
        queue = queue[:args.limit]
    todo = [p for p in queue if p["pair_id"] not in done(rater)]
    print(f"{rater} | {model_id} | {len(todo)} of {len(queue)} pairs (2 calls each, order-swapped)")
    if not todo:
        print("nothing to do; already rated this slice.")
        return

    t0 = time.time()
    print(f"loading {model_id} ...", flush=True)
    proc, model = load_model(model_id)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    tally = {"L": 0, "R": 0, "tie": 0, "incons": 0}
    for i, pair in enumerate(todo, 1):
        seg_l, seg_r, img_l, img_r, side_a = layout(pair)
        a1 = judge_once(proc, model, img_l, img_r)            # order A: seg_l on left
        a2 = judge_once(proc, model, img_r, img_l)            # order B: swapped
        p1 = picked(parse(a1), seg_l, seg_r)
        p2 = picked(parse(a2), seg_r, seg_l)                  # first image is seg_r here
        if p1 is not None and p1 == p2:                       # consistent across swap
            ch = "L" if p1 == seg_l else "R"
            winner, loser, cons = p1, (seg_r if p1 == seg_l else seg_l), 1
        elif parse(a1) == "tie" and parse(a2) == "tie":
            ch, winner, loser, cons = "tie", "", "", 1
        else:                                                 # flips with layout = biased
            ch, winner, loser, cons = "incons", "", "", 0
        tally[ch] += 1
        append({"ts_iso": dt.datetime.now().isoformat(timespec="seconds"), "rater": rater,
                "pair_id": pair["pair_id"], "pair_type": pair["pair_type"],
                "seg_left": seg_l, "seg_right": seg_r, "img_left": img_l, "img_right": img_r,
                "side_of_a": side_a, "choice": ch, "winner_seg": winner, "loser_seg": loser,
                "ans_order1": a1[:24], "ans_order2": a2[:24], "consistent": cons})
        if i % 4 == 0 or i == len(todo):
            el = time.time() - t0
            dec = tally["L"] + tally["R"]
            print(f"  {i}/{len(todo)}  {el:.0f}s  {el/i:.1f}s/pair  decisive={dec} "
                  f"tie={tally['tie']} incons={tally['incons']}  last=({a1[:8]!r},{a2[:8]!r})",
                  flush=True)
    dec = tally["L"] + tally["R"]
    n = len(todo)
    print(f"done: {rater} {n} pairs in {time.time()-t0:.0f}s -> {LOG.name}")
    print(f"decisive(order-consistent)={dec}/{n} ({dec/n:.0%})  tie={tally['tie']}  "
          f"incons(position-biased)={tally['incons']}  left-picks={tally['L']} right-picks={tally['R']}")


if __name__ == "__main__":
    main()

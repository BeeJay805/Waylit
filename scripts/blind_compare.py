"""Blind pairwise rating tool: which street feels safer to walk alone at night?

Local web app (Python standard library only, no install). For each seeded pair it shows two
DAYTIME Mapillary photos plus a downtown locator map, and asks the rater to pick. There are
NO model scores, no lighting numbers, no feature values anywhere on the page - the judgment
comes from the photo and rough location only. Choices append to data/ground_truth/pairwise.csv.
The tool is resumable (it serves the next unrated pair) and randomizes left/right per pair so
screen position never tracks a hidden variable.

Run:  python scripts/blind_compare.py            # then open http://127.0.0.1:8000
      python scripts/blind_compare.py --rater alex --port 8000
"""
import argparse
import csv
import datetime as dt
import html
import math
import pathlib
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "ground_truth"
IMGDIR = ROOT / "data" / "raw" / "mapillary" / "img"
QUEUE = GT / "pairwise_queue.csv"
LOG = GT / "pairwise.csv"
LOCATOR = PROC / "locator_base.png"
LOG_FIELDS = ["ts_iso", "rater", "pair_id", "pair_type", "seg_left", "seg_right",
              "img_left", "img_right", "side_of_a", "choice", "winner_seg", "loser_seg"]
IMG_RE = re.compile(r"^[0-9]+$")

CFG = {"rater": "rater1", "bbox": None, "queue": [], "allowed_imgs": set()}


def load_queue():
    with QUEUE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pair_id"] = int(r["pair_id"])
    CFG["queue"] = rows
    CFG["allowed_imgs"] = {r["img_a"] for r in rows} | {r["img_b"] for r in rows}


def done_pairs():
    if not LOG.exists():
        return set()
    with LOG.open(encoding="utf-8") as f:
        return {int(r["pair_id"]) for r in csv.DictReader(f) if r.get("pair_id")}


def append_choice(row):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_bbox():
    import yaml
    b = yaml.safe_load((ROOT / "config" / "area.yaml").read_text())["bbox"]
    CFG["bbox"] = (b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])


def ensure_locator():
    if LOCATOR.exists():
        return
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mnx, mny, mxx, mxy = CFG["bbox"]
    g = gpd.read_file(PROC / "segment_features.gpkg").to_crs("EPSG:4326")
    mean_lat = math.radians((mny + mxy) / 2)
    w = 540
    h = int(w * (mxy - mny) / ((mxx - mnx) * math.cos(mean_lat)))
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    g.plot(ax=ax, color="#c9ccd1", linewidth=0.5)
    ax.set_xlim(mnx, mxx); ax.set_ylim(mny, mxy)
    fig.savefig(LOCATOR, dpi=100)
    plt.close(fig)


def dot_pct(lon, lat):
    mnx, mny, mxx, mxy = CFG["bbox"]
    return (100 * (lon - mnx) / (mxx - mnx), 100 * (mxy - lat) / (mxy - mny))


def next_pair():
    done = done_pairs()
    for r in CFG["queue"]:
        if r["pair_id"] not in done:
            return r, len(done)
    return None, len(done)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Waylit blind rating</title><style>
*{{box-sizing:border-box}}body{{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0f1115;color:#e7e9ee}}
.bar{{padding:10px 16px;background:#171a21;display:flex;gap:16px;align-items:center;font-size:14px;color:#aab}}
.bar b{{color:#fff}}.wrap{{max-width:1080px;margin:0 auto;padding:16px}}
.q{{text-align:center;font-size:20px;font-weight:600;margin:6px 0 14px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.card{{background:#171a21;border-radius:10px;overflow:hidden;border:1px solid #232733}}
.card img{{width:100%;height:300px;object-fit:cover;display:block;background:#000}}
.tag{{text-align:center;padding:7px;font-weight:700;letter-spacing:.5px;color:#cdd}}
.btns{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-top:14px}}
button{{font-size:15px;padding:14px 8px;border-radius:9px;border:1px solid #2a2f3a;background:#222734;color:#fff;cursor:pointer}}
button:hover{{background:#2c3342}}.b-l{{border-color:#3b6ea5}}.b-r{{border-color:#c07a32}}
.note{{text-align:center;color:#7b8190;font-size:12.5px;margin-top:12px}}
.loc{{margin:16px auto 0;position:relative;width:540px;max-width:100%}}
.loc img{{width:100%;border-radius:8px;border:1px solid #232733;display:block}}
.dot{{position:absolute;width:14px;height:14px;border-radius:50%;transform:translate(-50%,-50%);border:2px solid #0f1115}}
.dot.l{{background:#4b8fd6}}.dot.r{{background:#e0903f}}
.legend{{text-align:center;font-size:12.5px;color:#9aa;margin-top:6px}}
.kbd{{color:#9aa;font-size:12px}}
</style></head><body>
<div class="bar"><b>Waylit</b> blind pairwise &nbsp;|&nbsp; rater <b>{rater}</b>
&nbsp;|&nbsp; progress <b>{done}</b> / {total}
&nbsp;|&nbsp; <span class="kbd">keys: F = left &nbsp; J = right &nbsp; T = same &nbsp; S = skip</span></div>
<div class="wrap">
<div class="q">Which street feels safer to walk alone at night?</div>
<div class="pair">
 <div class="card"><img src="/img/{img_left}" alt="left"><div class="tag">LEFT (F)</div></div>
 <div class="card"><img src="/img/{img_right}" alt="right"><div class="tag">RIGHT (J)</div></div>
</div>
<form method="POST" action="/choice" id="frm">
 <input type="hidden" name="pair_id" value="{pair_id}">
 <input type="hidden" name="pair_type" value="{pair_type}">
 <input type="hidden" name="seg_left" value="{seg_left}">
 <input type="hidden" name="seg_right" value="{seg_right}">
 <input type="hidden" name="img_left" value="{img_left}">
 <input type="hidden" name="img_right" value="{img_right}">
 <input type="hidden" name="side_of_a" value="{side_of_a}">
 <div class="btns">
  <button class="b-l" name="choice" value="L" type="submit">Left feels safer</button>
  <button name="choice" value="tie" type="submit">About the same</button>
  <button name="choice" value="skip" type="submit">Skip</button>
  <button class="b-r" name="choice" value="R" type="submit">Right feels safer</button>
 </div>
</form>
<div class="loc">
 <img src="/locator.png" alt="downtown locator">
 <div class="dot l" style="left:{lx}%;top:{ly}%"></div>
 <div class="dot r" style="left:{rx}%;top:{ry}%"></div>
</div>
<div class="legend"><span style="color:#4b8fd6">&#9679;</span> left photo &nbsp;&nbsp;
 <span style="color:#e0903f">&#9679;</span> right photo &nbsp; (rough downtown location)</div>
<div class="note">No safety scores or map data are shown. Judge from the photo and rough location only.
 Daytime photos are a stand-in for the night question - read the street, not the daylight.</div>
</div>
<script>
document.addEventListener('keydown',function(e){{
 var k=e.key.toLowerCase(),m={{f:'L',j:'R',t:'tie',s:'skip'}};
 if(m[k]){{var b=document.querySelector('button[value="'+m[k]+'"]');if(b)b.click();}}
}});
</script></body></html>"""

DONE = """<!doctype html><html><head><meta charset="utf-8"><title>Waylit done</title>
<style>body{{font-family:system-ui,Arial;background:#0f1115;color:#e7e9ee;text-align:center;padding:60px}}
b{{color:#7fd17f}}</style></head><body>
<h2>All {total} pairs rated. <b>Thank you.</b></h2>
<p>Logged to data/ground_truth/pairwise.csv</p>
<p>Tallies: {tally}</p>
<p>Next: run <code>python scripts/fit_pairwise_models.py</code> to calibrate weights and
compare structured vs visual vs fusion.</p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._root()
        if path == "/locator.png":
            return self._send(LOCATOR.read_bytes(), "image/png")
        if path.startswith("/img/"):
            return self._img(path[5:])
        self._send("not found", code=404)

    def _img(self, name):
        iid = name[:-4] if name.endswith(".jpg") else name
        if not IMG_RE.match(iid) or iid not in CFG["allowed_imgs"]:
            return self._send("forbidden", code=403)
        p = IMGDIR / f"{iid}.jpg"
        if not p.exists():
            return self._send("missing", code=404)
        self._send(p.read_bytes(), "image/jpeg")

    def _root(self):
        pair, done = next_pair()
        total = len(CFG["queue"])
        if pair is None:
            tally = ""
            if LOG.exists():
                import collections
                c = collections.Counter()
                with LOG.open(encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        c[r["choice"]] += 1
                tally = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
            return self._send(DONE.format(total=total, tally=html.escape(tally)))
        # stable left/right assignment per pair
        a_left = random.Random(pair["pair_id"] * 2654435761 & 0xFFFFFFFF).random() < 0.5
        if a_left:
            seg_l, seg_r = pair["seg_a"], pair["seg_b"]
            img_l, img_r = pair["img_a"], pair["img_b"]
            lon_l, lat_l = float(pair["lon_a"]), float(pair["lat_a"])
            lon_r, lat_r = float(pair["lon_b"]), float(pair["lat_b"])
            side_of_a = "L"
        else:
            seg_l, seg_r = pair["seg_b"], pair["seg_a"]
            img_l, img_r = pair["img_b"], pair["img_a"]
            lon_l, lat_l = float(pair["lon_b"]), float(pair["lat_b"])
            lon_r, lat_r = float(pair["lon_a"]), float(pair["lat_a"])
            side_of_a = "R"
        lx, ly = dot_pct(lon_l, lat_l)
        rx, ry = dot_pct(lon_r, lat_r)
        self._send(PAGE.format(
            rater=html.escape(CFG["rater"]), done=done, total=total,
            pair_id=pair["pair_id"], pair_type=html.escape(pair["pair_type"]),
            seg_left=seg_l, seg_right=seg_r, img_left=img_l, img_right=img_r,
            side_of_a=side_of_a, lx=round(lx, 2), ly=round(ly, 2),
            rx=round(rx, 2), ry=round(ry, 2)))

    def do_POST(self):
        if urlparse(self.path).path != "/choice":
            return self._send("not found", code=404)
        n = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode("utf-8")).items()}
        pid = int(form.get("pair_id", -1))
        if pid not in done_pairs():  # idempotent: ignore double-submit of same pair
            choice = form.get("choice", "skip")
            sl, sr = form.get("seg_left", ""), form.get("seg_right", "")
            winner = sl if choice == "L" else sr if choice == "R" else ""
            loser = sr if choice == "L" else sl if choice == "R" else ""
            append_choice({
                "ts_iso": dt.datetime.now().isoformat(timespec="seconds"),
                "rater": CFG["rater"], "pair_id": pid,
                "pair_type": form.get("pair_type", ""),
                "seg_left": sl, "seg_right": sr,
                "img_left": form.get("img_left", ""), "img_right": form.get("img_right", ""),
                "side_of_a": form.get("side_of_a", ""), "choice": choice,
                "winner_seg": winner, "loser_seg": loser,
            })
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rater", default="rater1")
    args = ap.parse_args()
    CFG["rater"] = args.rater
    load_queue()
    read_bbox()
    ensure_locator()
    _, done = next_pair()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Waylit blind rating | rater={args.rater} | {done}/{len(CFG['queue'])} done")
    print(f"open  http://127.0.0.1:{args.port}   (Ctrl+C to stop; progress is saved)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. progress saved to data/ground_truth/pairwise.csv")


if __name__ == "__main__":
    main()

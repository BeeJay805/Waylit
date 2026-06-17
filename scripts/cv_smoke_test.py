"""CV smoke test: confirm the visual-feature pipeline works on this CPU and time it.

Loads a small segmentation model (SegFormer-b0, Cityscapes), runs it on a sample of the
downloaded Mapillary images, reports per-image speed + mean class fractions, and saves a
few segmentation visualizations for a sanity check. No labels needed.

Run:  python scripts/cv_smoke_test.py
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "raw" / "mapillary" / "img"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = 40
MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
CLASSES = {0: "road", 1: "sidewalk", 2: "building", 5: "pole", 8: "vegetation", 10: "sky"}
PALETTE = {0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
           5: (153, 153, 153), 8: (107, 142, 35), 10: (70, 130, 180)}


def log(m):
    print(m, flush=True)
    with (OUT / "cv_smoke.log").open("a", encoding="utf-8") as f:
        f.write(m + "\n")


def ensure_deps():
    try:
        import torch  # noqa
        import transformers  # noqa
        from PIL import Image  # noqa
    except Exception:
        log("installing torch/transformers/pillow (one-time) ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch", "transformers", "pillow"], check=False)


def main():
    ensure_deps()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    paths = sorted(IMG.glob("*.jpg"))[:SAMPLE]
    if not paths:
        log("no images found"); return
    log(f"loading {MODEL} ...")
    proc = AutoImageProcessor.from_pretrained(MODEL, size={"height": 512, "width": 512})
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    def infer(p):
        img = Image.open(p).convert("RGB")
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inp).logits[0]
        return img, logits.argmax(0).cpu().numpy()

    times, fr = [], {k: [] for k in CLASSES}
    for i, p in enumerate(paths):
        t = time.time()
        img, pred = infer(p)
        times.append(time.time() - t)
        tot = pred.size
        for cid in CLASSES:
            fr[cid].append(float((pred == cid).sum()) / tot)
        if i < 3:
            up = np.array(img.resize((pred.shape[1], pred.shape[0])))
            mask = np.zeros_like(up)
            for cid, rgb in PALETTE.items():
                mask[pred == cid] = rgb
            blend = (0.5 * up + 0.5 * mask).astype("uint8")
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(up); ax[0].set_title("image"); ax[0].axis("off")
            ax[1].imshow(blend); ax[1].set_title("segmentation"); ax[1].axis("off")
            fig.savefig(OUT / f"cv_sample_{i + 1}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(paths)}  mean {sum(times) / len(times):.2f}s/img")

    per = float(np.median(times))
    summary = {"images": len(paths), "median_s_per_image": round(per, 2),
               "projected_hours_for_30k": round(per * 30000 / 3600, 1),
               "mean_fractions": {CLASSES[k]: round(float(np.mean(fr[k])), 3) for k in CLASSES}}
    (OUT / "cv_smoke_summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()

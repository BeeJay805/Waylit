# Phase 0: Mapillary imagery coverage report (downtown Boise)

Generated 2026-06-15 by `scripts/mapillary_coverage.py` over the study bbox
(`-116.220, 43.600, -116.185, 43.630`), using an 8x8 grid (64 cells, ~350 m each).

## Numbers (live scan)

- Coverage: 50 of 64 cells have imagery = **78% of the area** (cell-level proxy).
- Density: median ~28 images per covered cell; 8 cells hit the 50-image sample cap (dense).
- Images sampled: 1,281 (true total is higher; sampling capped at 50 per cell).
- Capture years: 2018 (2), 2019 (165), 2020 (752), 2023 (362). Newest sampled: May 2023.
- Panoramas: 0% (all perspective cameras).
- Viewing directions: all 8 compass octants present (streets seen from multiple directions).

## What the images actually look like

Four samples were pulled (saved in `data/raw/mapillary_samples/`). They confirm the
imagery is usable for the perception tasks, and they surface three quality caveats:

- A downtown office block with a decorative streetlamp, building frontage, and trees:
  exactly the lamp + enclosure + frontage signal we need.
- A residential intersection: clear road geometry, crossings, mature trees, a distant lamp.
- A greenbelt path: path surface and heavy vegetation.
- A wider arterial near the bbox edge: road, sky, utility pole.

Caveats visible in the images:

1. **Mixed camera positions.** Some are dashcam shots from a car windshield (hood and
   glass glare at the edges, taken from the road centreline), others are pedestrian or
   bike level. Sidewalk-scale features read better from the pedestrian shots, so the
   pipeline should record and possibly filter by camera position.
2. **Season matters a lot.** Some shots are leaf-off winter, others leaf-on summer. A
   bare winter tree hides little; the same tree in summer can block a lamp. Capture
   season must be tracked, and leaf-on imagery preferred for the canopy-occlusion feature.
3. **Age.** Most imagery is 2020 with a 2023 refresh. Fine for stable built features
   (lamps, buildings, road width); flag for anything that may have changed since.

## Verdict: gate PASSES, with caveats

There is enough usable daytime imagery to build the AI perception layer for downtown
Boise. The ~22% of cells with no imagery, plus within-cell gaps, are exactly where the
structured fallback and the simulate-missingness test matter. Coverage is uneven, so we
must audit for coverage bias (whether imaged streets are systematically different).

## Next (once the walk graph exists)

- DONE: clean segment-level coverage is 50.4% of walkable segments (image within 20 m),
  median 6 images where present; ~90% of all imagery is leaf-on (good for canopy occlusion).
- Record per image: camera position, season, capture date, compass angle, as covariates.
- Decide an imagery-usability filter (min images per segment, perspective vs pano, leaf-on).

# Phase 0: prior-art matrix

The novelty of Waylit is a hypothesis until this review is complete. For each close
piece of work, this states exactly what it used and produced, and what (if anything)
Waylit adds. Cells marked (abs) come from an abstract or summary and must be confirmed
against the full text before any publication claim.

| Work | Inputs | Target | Ground truth | Model | Geography | Output | What Waylit adds |
|---|---|---|---|---|---|---|---|
| Place Pulse 2.0 / Streetscore (MIT) | daytime Street View | perceived safety (+5 other perceptions) | ~1.2M online pairwise votes | CNN ranking | 56 cities, global | perceived-safety scores | physical night features (not daytime perception), measured night lux, calibration, routing |
| SafetiPin | night drive-by photos + raters | 9-parameter safety score | trained human auditors | rubric + some ML (abs) | Delhi + many cities | location safety scores | no original night imagery, open reproducible dataset, calibration, US downtown |
| Google "well-lit routes" | Street View (proprietary) | lit-street detection | not disclosed | proprietary | India pilot | in-app route highlight | open method + dataset, uncertainty, independent validation |
| Day-to-night generation, 24h safety (HK) | daytime SVI | synthetic night image + perceived safety | ~1,000 paired day/night + survey | GAN/diffusion + VLM | Hong Kong | 24h perceived-safety maps | discriminative (no synthesis), measured night lux, physical not perceptual, calibration |
| Nighttime SVI for lighting landscape | actual night SVI | lighting-landscape metrics | the night imagery (abs) | CV (abs) | confirm | lighting maps | infer potential from DAYTIME + inventory, validate vs measured lux, no night-imagery collection |
| OSM + SVI speed-class imputation (HeiGIT) | OSM + Street View | traffic speed class (structural) | not stated | not stated | Berlin | citywide speed-class map | night-grounded physical target, new ground truth, calibration, active learning |
| UrbanVGGT sidewalk width | Street View | sidewalk width (metres) | reference widths | VGGT geometry | Washington DC | per-segment width | night-comfort target; width is a borrowable feature; calibration |
| SAGAI (VLM streetscape) | SVI + vision-language model | various streetscape ratings | mostly VLM zero-shot | VLM | varies | mapped scores | VLM only as one feature estimator validated against measured ground truth, not the final score |

## The niche this leaves (still a hypothesis)

Combine, in one open project: a physical nighttime-walking target (lighting potential
mapped to measured illumination), grounded in new measured night data, with calibrated
uncertainty and provenance, active learning to spend a tiny audit budget well, and a
clean A / B1 / B2 / C comparison, in a data-rich US downtown where a strong structured
baseline makes the test honest. None of the works above combine these.

## What would change the claim

If a paper already maps daytime-derived lighting potential to measured night lux with
calibration, the novelty narrows to the active-learning, open-dataset, and routing
contributions. We keep checking before claiming anything, and we cite the closest work
above either way.

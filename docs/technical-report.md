# Waylit technical report (draft)

Downtown Boise, Idaho. Status: method and dataset built and partially validated; the
headline validation (against measured night conditions) is pending a field pilot.

## 1. What this is

Waylit estimates the physical and situational conditions that affect walking alone at
night, per street segment, from public data, and routes on them so a walker can choose a
calmer route and see why. It is built as an open dataset plus a transparent router, not a
consumer app, and it does not claim any route is objectively "safe."

## 2. Principles (held throughout)

- Environment-based, not crime-based. Crime and demographics are validation and context
  only, never model inputs (Section 6 shows why, empirically).
- The score must be reproducible and auditable. Weights are calibrated from human
  judgment, never hand-set.
- Communicate uncertainty and provenance. Distinguish observed from inferred. Never
  manufacture false confidence.
- Separate *lighting potential* (inferable from daytime data) from *actual night
  illumination* (only knowable by night measurement).

## 3. Study area and data

Downtown Boise, WGS84 bbox -116.220, 43.600, -116.185, 43.630. Sources verified live:

- City of Boise streetlight inventory: 13,965 lamps citywide, ~3,091 downtown, with
  per-lamp wattage, height, fixture, lamp type, and owner, ~98-100% complete across all
  owners including Idaho Power.
- OpenStreetMap: walk graph and dense sidewalk geometry (lighting tags are near-useless,
  so lighting comes from the inventory).
- Overture Maps: building footprints with heights, and places (businesses). National,
  open, monthly, which is the backbone for scaling.
- Valley Regional Transit GTFS: 562 stops, service to midnight, 214 night-active stops.
- Mapillary daytime street imagery: ~55k images in the bbox, ~50% segment coverage,
  ~90% leaf-on.
- Crime (BPD) and ACS income: validation and bias-audit only.

## 4. Method (four layers)

1. Public structured data (above).
2. AI perception and inference: daytime imagery to a frozen SegFormer (Cityscapes)
   segmentation model, giving per-segment fractions of sidewalk, vegetation, sky,
   building, and pole. Fast on CPU (~0.22 s/image).
3. Human ground truth and active learning (pilot, pending).
4. Deterministic scoring and routing: explicit, adjustable, auditable weights; route
   explanations; never claims "safe."

Lighting model (System A, deterministic): for points sampled along a segment, sum a
point-source illuminance proxy from nearby lamps,
`E ~ Wattage * H / (d^2 + H^2)^1.5`, within 35 m. Auditable, no learning.

Tiered design for scale: each feature ranks its sources and records which it used.
Lighting is Tier 1 inventory where available, then OSM tags, then imagery-inferred
potential. Most features rely on nationally available sources (Overture, OSM, GTFS).

## 5. The planned experiment (gated on the pilot)

Four comparable systems on the same measured night labels:
A structured-only, B1 frozen visual features, B2 adapted visual encoder, C multimodal
fusion. Metrics: feature coverage, agreement with night audits, calibrated uncertainty,
geographic bias, missing-data performance, and downstream route-ranking agreement.
A field pilot of 30-50 night-audited segments (protocol in `docs/phase0/`) seeds this.

## 6. Findings so far

- **Imagery recovers lighting only modestly.** Predicting the structured lighting score
  from imagery on spatially held-out segments gives Spearman ~0.39, R2 ~0.20, mostly via
  urban form (building +0.36, vegetation -0.36), not direct lamp detection. So imagery is
  a rough fallback where no inventory exists, not a substitute for one.
- **Crime and poverty carry no usable street-level signal.** On night, outdoor,
  person-crimes: correlation with poverty +0.07; spatial-CV R2 from environment ~0 and
  from environment+poverty ~0.03. The one strong correlation, lighting vs crime +0.48, is
  a downtown-activity confound that does not survive spatial cross-validation. Feeding
  crime into routing would steer walkers toward dark, empty streets. Hence crime stays
  validation-only.
- **The feature set is clean of demographic bias.** Correlation of each feature with
  tract low-income share is under 0.2 for all features; lighting is 0.00. The features are
  environmental, not demographic proxies. (Caveat: tract-level, 14 tracts, downtown only.)
- **Routing trade-off is real and small.** A calmer multi-factor route (lighting plus
  business activity, provisional weights) costs about +1.8 minutes for a comfort gain
  from 0.36 to 0.58.

## 7. Limitations (read this)

- Not validated against measured night conditions. The pilot is the headline gate.
- Provisional routing weights; no calibrated safety score is published yet, by design.
- One city, one downtown. Generalization is untested.
- Reported-crime location is a confounded proxy, not actual risk or perceived safety.
- Imagery: mixed camera positions and seasons, ~50% segment coverage, ~25% after the
  partial download.
- Bias audit is tract-coarse and downtown-only; weak correlations may differ at finer
  resolution or in other cities.

## 8. Scaling plan (Metro US, later)

Per-metro config feeding one pipeline on the Overture backbone (buildings, places,
transportation), with a small streetlight-inventory adapter registry where cities publish
one, Mapillary imagery, and GTFS via the Mobility Database. Tiered features with per-
segment uncertainty and provenance. Compute is not the bottleneck once imagery is sampled
to a few per segment; validation and inventory heterogeneity are. Plan: deep-validate a
few anchor metros with pilots, rely on calibrated uncertainty and honest "validated in N
cities" labeling elsewhere.

## 9. Reproducibility

All results regenerate from `scripts/` plus `config/area.yaml`: `probe_structured_sources`,
`build_system_a`, `mapillary_coverage`, `extract_visual_features`, `build_features`,
`add_transit_feature`, `route_demo[_v2]`, `exp_imagery_predicts_lighting`,
`exp_crime_vs_environment`, `add_poverty_analysis`, `bias_audit_features`, `export_dataset`.

## 10. Next steps

1. Field pilot (30-50 night-audited segments).
2. Calibrate weights and run the A/B1/B2/C spike with calibrated uncertainty.
3. Publish dataset v1 with a validated, calibrated comfort score and uncertainty.
4. Outreach to a campus safety office or women's safety organization.
5. Begin the multi-city scale phase.

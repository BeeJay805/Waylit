# Waylit technical report

Downtown Boise, Idaho, with a Los Angeles validation transect. Status: an open dataset plus a
transparent router, with two per-segment scores - a **night-comfort** score calibrated from blind
pairwise human judgments (no night walk required) and an **objective, crash-validated day-safety**
score. Both are honest bootstraps with documented limits; the night score still needs multiple
raters and measured night lux to move past single-rater calibration.

## 1. What this is

Waylit estimates the physical and situational conditions that affect **walking**, per street
segment, from public data, and routes on them so a walker can choose a calmer or safer route and
see why. The dataset is built on the pedestrian (walk) network, not the car network. It carries
two distinct scores because day and night risks differ: by **day** the hazard is traffic
(separation from cars, speed); by **night** it is comfort (lighting, activity, transit). It is an
open dataset plus a transparent router, not a consumer app, and it does not claim any route is
objectively "safe."

## 2. Principles (held throughout)

- Environment-based, not crime-based. Crime and demographics are validation and context
  only, never model inputs (Section 6 shows why, empirically).
- Reproducible and auditable. Night-comfort weights are calibrated from human judgment; the
  day-safety score is grounded in established pedestrian-safety evidence. Neither is arbitrarily
  hand-set, and both carry documented limits.
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
3. Human ground truth via **blind pairwise** judgments, no night walk required. A local tool
   shows two daytime photos with no model scores and asks "which feels safer to walk alone at
   night?"; the choices fit a transparent logistic learning-to-rank that calibrates the night
   weights. (Daytime photos are an honest proxy here; see the limitation in Section 6/7.)
4. Deterministic scoring and routing: explicit, learned/auditable weights; route explanations;
   never claims "safe."

Lighting model (System A, deterministic): for points sampled along a segment, sum a
point-source illuminance proxy from nearby lamps,
`E ~ Wattage * H / (d^2 + H^2)^1.5`, within 35 m. Auditable, no learning.

**Day-safety layer (objective).** The night features miss the dominant *daytime* hazard, traffic.
A pedestrian-traffic layer from OSM (car-free vs road class, posted speed, lanes, sidewalk presence,
crossings) feeds a transparent `day_safety` score reasoned from established pedestrian-safety
relationships: separation from cars, the speed-fatality curve (logistic near 35 mph), and sidewalk
protection. Day safety is objective enough to score without a subjective rating, and is validated
against real pedestrian crashes (Section 6).

Tiered design for scale: each feature ranks its sources and records which it used.
Lighting is Tier 1 inventory where available, then OSM tags, then imagery-inferred
potential. Most features rely on nationally available sources (Overture, OSM, GTFS).

## 5. Calibrating the night score without a night walk

Rather than wait on a night-audit pilot, we calibrated from **blind pairwise** judgments. One rater
made 150 "which feels safer to walk alone at night?" calls on daytime-photo pairs spanning the
lighting and business range across downtown (120 decisive, 21 ties, a clean 60/60 left-right split).
A logistic Bradley-Terry model on signed feature differences learned the weights, and we compared
three systems with spatial-block cross-validation: A structured-only, B1 frozen visual features,
C fusion.

Result: lighting (+0.42) and night transit (+0.38) are the significant learned drivers. Imagery
does **not** help: across 2-5 spatial blocks A/B1/C all sit ~0.44-0.61 with wide, overlapping
intervals, no evidence B1 or fusion beats the structured baseline. The robust, leak-free signal is
that the rater picks the brighter, more active street ~71% of the time.

Honest reframing (from the rater): those picks were really "how central, nice, and walkable does it
look" from the daytime photo, not visible night brightness, and Boise has little real danger
variance. So here lighting proxies central-and-nice. The higher-variance city in Section 6 shows
the durable driver is activity/business density, not lighting.

## 6. Findings

- **Imagery recovers lighting only modestly, and adds nothing to the night score.** Predicting the
  lighting score from imagery on spatially held-out segments gives Spearman ~0.39 (mostly urban
  form, not lamp detection), and in the calibration (Section 5) frozen-visual and fusion models do
  not beat the structured baseline. Imagery is a rough fallback where no inventory exists, not a
  substitute.
- **Crime carries no usable, unconfounded signal, and is exposure-driven.** On night/outdoor/
  person-crimes the environment explains ~0 out of sample; lighting-vs-crime +0.48 is a downtown-
  activity confound. In a higher-variance city (LA) comfort correlates *positively* with crime
  counts (+0.30) because busy, lit blocks have more people; even after controlling for ambient
  population (ACS residents + LODES jobs) lighting still tracks more crime (+0.23). Free data cannot
  show comfort = lower risk, and routing on crime would steer walkers onto dark, empty streets.
  Crime stays validation-only.
- **The features are clean of demographic bias, including in a diverse city.** In Boise every
  feature correlates < 0.2 with tract low-income share (lighting 0.00). In a Central/South LA
  transect (income $15k-$107k, racially diverse) the comfort features stay mostly clean: lighting is
  cleanest (|r| <= 0.03 vs income and % nonwhite), the largest is building height vs % nonwhite at
  -0.15. The learned comfort score has a mild, borderline tilt (+0.06 income, -0.10 % nonwhite),
  traced to valuing business density; worth monitoring, not a strong proxy.
- **Activity, not lighting, is the durable night driver.** With lighting's variance removed (LA is
  uniformly lit), the rater's "safer" calls track business/activity density most (+0.56, 71%); the
  Boise lighting weight was largely lighting-as-centrality.
- **Day-safety ordering is validated by real crashes.** Among roads, pedestrian-vehicle crashes
  (COMPASS/ITD, 108 in the bbox) concentrate on low-`day_safety` segments: Spearman -0.21, counts
  rising by road class (residential 6, tertiary 13, secondary 19). This validates the ordering
  (arterials are more dangerous to walk by day), not the exact constants.
- **Routing trade-off is real and small.** With the calibrated night weights, a calmer route costs
  about +1.8 minutes (+6%) for a comfort gain from 0.57 to 0.78.

## 7. Limitations (read this)

- **Night score is a single-rater bootstrap.** One rater, daytime photos as a night proxy, not the
  target demographic (women walking alone). The rater's picks were partly an aesthetic/centrality
  judgment, not visible night brightness. Final calibration needs multiple raters and measured lux.
- **Day score uses reasoned constants.** `day_safety` is validated for *ordering* against real
  crashes, but the exact constants (e.g. the 35 mph fatality midpoint) are literature-grounded, not
  refit; sidewalk presence is a Boise data gap (1% tagged), so it leans on separation + speed.
- **Crime is a confounded proxy.** Exposure-driven and demographically entangled; never an input.
- Two cities (Boise plus an LA transect); per-city validation is still thin.
- Imagery: mixed camera positions and seasons, ~25% segment coverage.
- The LA bias audit is block-group-level on one transect; the mild comfort tilt may differ elsewhere.

## 8. Scaling plan (Metro US, later)

Per-metro config feeding one pipeline on the Overture backbone (buildings, places,
transportation), with a small streetlight-inventory adapter registry where cities publish
one, Mapillary imagery, and GTFS via the Mobility Database. Tiered features with per-
segment uncertainty and provenance. Compute is not the bottleneck once imagery is sampled
to a few per segment; validation and inventory heterogeneity are. Plan: deep-validate a
few anchor metros with pilots, rely on calibrated uncertainty and honest "validated in N
cities" labeling elsewhere.

## 9. Reproducibility

All results regenerate from `scripts/` plus `config/`. Feature build: `build_features`,
`add_transit_feature`, `build_walk_safety`, `score_day_safety`. Night calibration:
`build_pairwise_link`, `seed_pairs`, `blind_compare`, `fit_pairwise_models`. Routing:
`route_calibrated`. Validation: `exp_crime_vs_environment`, `bias_audit_features`, `validate_city`,
`exposure_normalize`, `validate_day_safety`. Portability: `build_city_features`, `fetch_la_rating`,
`fit_la_rating`. Release: `export_dataset` (v1, both scores).

## 10. Next steps

1. **Multiple night raters**, especially women who walk alone; then refit and report inter-rater
   agreement (the rating tool is already multi-rater and day/night-mode aware).
2. Measured night lux on a stratified pilot to ground the lighting feature and the daytime proxy.
3. A day-safe router to match the night router; richer sidewalk data where cities publish it.
4. Outreach to a campus safety office or a women's safety organization.
5. Continue the multi-city scale phase, with the per-city bias audit as a gate.

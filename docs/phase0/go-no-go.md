# Phase 0: go / no-go memo (living draft)

Decision: proceed with Boise as planned, adjust scope, or switch city. This fills in as
evidence arrives; it is not final yet.

## Decision criteria

1. Does the core structured data (streetlights) exist and is it usable?
2. Is there usable daytime imagery coverage in the study area?
3. Can a solo student collect enough night ground truth at acceptable burden?
4. Is there a defensible research question once prior art is accounted for?
5. Expected compute cost stays within a small budget.

## Evidence so far

1. Structured data: STRONG. Streetlight inventory verified (13,965 citywide, ~3,091
   downtown, ~98-100% complete photometric fields across all owners incl Idaho Power).
   Buildings carry height; trees carry species + trunk diameter. See
   structured-data-availability.md.
2. Imagery coverage: PASS (with caveats). 78% of downtown grid cells have daytime
   imagery, median ~28 per cell, mostly 2020 with a 2023 refresh, all perspective, all
   compass directions, and visibly usable for lamps/enclosure/frontage. Caveats: mixed
   camera positions, season affects canopy occlusion, ~22% of cells uncovered. See
   coverage-report.md.
3. Ground-truth burden: UNKNOWN until the pilot runs (protocol is locked).
4. Research question: a defensible niche exists as a hypothesis (see prior-art.md), but
   it has shifted. Because Boise's structured data is rich, the question is the marginal
   value of imagery + adaptation over a strong baseline, tested via simulate-missingness.
5. Compute: low so far (all checks are stdlib API calls). The AI layer adds
   pretrained-inference plus small-head training, at laptop or cheap-GPU scale.

## Open questions for the decision

- Is Mapillary coverage in downtown Boise dense and recent enough to support the AI layer?
- Given the strong baseline, is "marginal value of imagery" interesting enough alone, or
  do we add a second, data-poorer city for a transfer test rather than relying only on
  simulate-missingness?
- What pilot result would justify the B2 (fine-tuning) arm? Proposed bar: frozen features
  (B1) leave a clear, consistent error against the night labels that adaptation reduces
  on spatially held-out segments, beyond noise.

## Provisional lean

Boise is an excellent place to VALIDATE the method and study the potential-to-measured
relationship. Its data richness is a strength for rigor and a challenge for showcasing
imagery's gap-filling. The imagery coverage gate has passed (78% of cells, with
caveats); the remaining input is the night pilot.

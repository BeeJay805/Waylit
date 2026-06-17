# Waylit open dataset, v1: downtown Boise walk segments

`waylit_boise_downtown_v1.csv`, one row per walkable street segment in the downtown study
area (WGS84 bbox -116.220, 43.600, -116.185, 43.630). 3,747 segments. The network is the
pedestrian (walk) graph: 81% dedicated car-free ways (footways, paths, plazas) and 19% road
centerlines walked alongside; car-only roads are excluded.

**Two transparent scores, both bootstrap, not final.** Unlike v0 (features only), v1 includes:

- `comfort_night` (0-1): how comfortable a segment feels to walk **alone at night**. Weights are
  LEARNED from one rater's blind pairwise judgments, never hand-set. Bootstrap: a single rater
  using daytime photos as a proxy; not the target demographic. Lighting and night transit are the
  significant learned drivers (but in low-danger-variance Boise, lighting partly proxies central-
  and-nice; a higher-variance city showed activity/business is the durable driver).
- `day_safety` (0-1): how safe a segment is to walk **by day**, where the dominant hazard is
  traffic. Reasoned from established pedestrian-safety relationships (separation from cars, the
  speed-fatality curve, lanes, sidewalks), NOT learned from a rater (day safety is objective).
  Validated: among roads, real pedestrian crashes concentrate on the low-`day_safety` segments
  (Spearman -0.21, crashes rise with road class). Validates the ordering, not the exact constants.

Crime and demographics are validation/context only, never inputs to either score.

## Columns

| Column | Meaning | Source | Type | Units |
|---|---|---|---|---|
| `seg_id` | Segment id (stable within this build) | derived | id | - |
| `u`, `v`, `key` | OSM node ids + edge key | OSM | id | - |
| `street`, `highway` | Street name, OSM way class | OSM | text | - |
| `length_m` | Segment length | OSM | observed | metres |
| `lat`, `lon` | Segment midpoint | derived | observed | WGS84 |
| `light` | Modeled lighting potential (lamp illuminance proxy, wattage+height weighted) | Boise streetlight inventory | observed (public) | proxy, higher=more lit |
| `encl_frontage` | Fraction of segment with a building within 20 m | Overture buildings | observed (public) | 0-1 |
| `encl_height` | Mean nearby building height | Overture | observed (public) | metres |
| `poi_density` | Businesses within 50 m | Overture places | observed (public) | count |
| `poi_night_density` | Night-active-category businesses within 50 m (category proxy) | Overture places | observed (public) | count |
| `transit_night_400m` | Transit stops with evening service (>=20:00) within 400 m | VRT GTFS | observed (public) | count |
| `vis_sidewalk` `vis_vegetation` `vis_sky` `vis_building` `vis_pole` | Mean pixel fraction from daytime street-image segmentation | Mapillary + SegFormer | inferred (imagery) | 0-1, null where `has_visual`=0 |
| `has_visual` | Imagery-derived features present | derived | flag | 0/1 |
| `car_free` | Dedicated pedestrian way (footway/path/pedestrian), full separation from cars | OSM | observed | 0/1 |
| `traffic_class` | Road class as traffic exposure: 0 car-free .. 5 primary arterial | OSM | observed | ordinal |
| `maxspeed_mph`, `lanes` | Posted speed and lane count (class default where untagged) | OSM | observed/derived | mph, count |
| `traffic_exposure` | Exposure to vehicle traffic: 0 car-free .. ~1 busy arterial | derived (OSM) | 0-1 |
| `sidewalk_present` | 1 car-free or OSM `sidewalk` present; 0 absent; 0.5 unknown (Boise tags ~1% of roads) | OSM | observed/flag | 0/0.5/1 |
| `crossing_near` | Marked/signalized crossings within 40 m | OSM | observed | count |
| **`day_safety`** | **Day-walking safety (objective, traffic-based; crash-validated ordering)** | derived | **score** | **0 dangerous .. 1 safe** |
| **`comfort_night`** | **Night-walking comfort (learned from blind pairwise; bootstrap)** | derived | **score** | **0 .. 1, higher = more comfortable** |
| `geometry_wkt` | Segment geometry | OSM | observed | WGS84 LineString (WKT) |

## Provenance and confidence

"observed (public)" features come from public records. "inferred (imagery)" features are model
estimates from daytime photos (~25% coverage). `light` estimates built lighting *potential*, not
measured night brightness. `day_safety` is reasoned from documented pedestrian-safety relationships
(auditable constants) and validated against real crashes for ordering. `comfort_night` is learned
from a single rater's blind judgments and is the least settled column.

## Known limitations

- `comfort_night`: one rater, daytime photos as a night proxy, not the target demographic (women
  walking alone). A bootstrap; final calibration needs multiple raters and measured night lux.
- `day_safety`: reasoned constants (e.g. 35 mph fatality midpoint) validated for ordering only;
  sidewalk presence is a Boise data gap (1% tagged), so the score leans on separation + speed.
- Crime is exposure-confounded and demographically entangled; even after controlling for ambient
  population it could not show comfort = lower risk. Hence crime stays validation-only.
- Downtown Boise; the pipeline is portable but per-city validation is thin.

## License and attribution

Derived from OpenStreetMap (ODbL), Overture Maps / Microsoft Building Footprints (ODbL), City of
Boise streetlights (CC-BY 4.0), Valley Regional Transit GTFS (CC-BY 3.0), and Mapillary imagery
(CC-BY-SA). Pedestrian-crash validation uses COMPASS/ITD crash data (validation only, not
redistributed). Released under **ODbL** with attribution to all sources; share-alike applies.

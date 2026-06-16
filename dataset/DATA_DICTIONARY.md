# Waylit open dataset, v0 (DRAFT): downtown Boise walk segments

`waylit_boise_downtown_v0.csv`, one row per walkable street segment in the downtown
study area (WGS84 bbox -116.220, 43.600, -116.185, 43.630). 3,747 segments.

**This is a DRAFT and is NOT a safety score.** It is a table of environmental *features*
per segment. We deliberately do not publish a combined "safety" or "comfort" score yet,
because the weights must be calibrated from human night judgments (the pilot), not hand-set.
None of these features has been validated against measured night conditions yet.

## Columns

| Column | Meaning | Source | Type | Units |
|---|---|---|---|---|
| `seg_id` | Segment id (stable within this build) | derived | id | - |
| `u`, `v`, `key` | OSM node ids + edge key (join back to OSM) | OSM | id | - |
| `street` | Street name (may be blank) | OSM | text | - |
| `highway` | OSM road class | OSM | text | - |
| `length_m` | Segment length | OSM | observed | metres |
| `lat`, `lon` | Segment midpoint | derived | observed | WGS84 |
| `light` | Modeled lighting potential (sum of lamp illuminance proxy, wattage and height weighted) | City of Boise streetlight inventory | **observed (public)** | unitless proxy, higher = more lit |
| `encl_frontage` | Fraction of segment with a building within 20 m | Overture buildings | observed (public) | 0-1 |
| `encl_height` | Mean nearby building height | Overture / MS Buildings | observed (public) | metres |
| `poi_density` | Businesses within 50 m | Overture places | observed (public) | count |
| `poi_night_density` | Night-active-category businesses within 50 m (category proxy; hours unavailable) | Overture places | observed (public) | count |
| `transit_night_400m` | Transit stops with evening service (departure hour >= 20) within 400 m | VRT GTFS | observed (public) | count |
| `vis_sidewalk` `vis_vegetation` `vis_sky` `vis_building` `vis_pole` | Mean pixel fraction from daytime street-image segmentation | Mapillary + SegFormer (Cityscapes) | **inferred (imagery)** | 0-1, null where `has_visual`=0 |
| `has_visual` | Whether imagery-derived features are present | derived | flag | 0/1 |
| `geometry_wkt` | Segment geometry | OSM | observed | WGS84 LineString (WKT) |

## Provenance note

"observed (public)" features come directly from public records. "inferred (imagery)"
features are model estimates from daytime photos and carry more uncertainty; they cover
only ~25% of segments (where usable imagery exists). The lighting feature estimates
*built lighting potential*, not measured night brightness.

## Known limitations

- Not validated against measured night conditions (pilot pending).
- Downtown Boise only; one city.
- Imagery features: mixed camera positions and seasons; ~25% segment coverage.
- `poi_night_density` is a category proxy (opening hours are not available).
- No calibrated safety/comfort score is included by design.

## License and attribution

Derived from OpenStreetMap (ODbL), Overture Maps / Microsoft Building Footprints (ODbL),
City of Boise streetlights (CC-BY 4.0), Valley Regional Transit GTFS (CC-BY 3.0), and
Mapillary imagery (CC-BY-SA). The combined dataset is released under **ODbL** with
attribution to all sources. Share-alike applies.

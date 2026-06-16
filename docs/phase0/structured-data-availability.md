# Phase 0: structured-data availability matrix (Boise downtown)

Footprint: WGS84 bbox `-116.220, 43.600, -116.185, 43.630` (downtown + buffer).
"Verified" means the endpoint was queried live on 2026-06-15. ArcGIS org base:
`https://services1.arcgis.com/WHM6qC35aMtyAAlN/arcgis/rest/services`.

| Source | Service / access | Role | Coverage | Update | License | Verified | Missingness / notes |
|---|---|---|---|---|---|---|---|
| Streetlight inventory | `Boise_Streetlights_Open_Data/FeatureServer/0` | Primary lighting (A) | 13,965 citywide, 3,091 in bbox | continual (modified Jun 2026) | CC-BY-4.0 | Yes (deep) | ~98-100% complete on wattage/height/fixture/lamp/Kelvin across all owners incl Idaho Power; `Retired_Date` filters inactive |
| Street trees | `BPR_Park_And_Street_Trees/FeatureServer/0` | Canopy occlusion (structured) | citywide | unknown | CC-BY-4.0 | Schema seen | has Species, Tree_Diameter (trunk), ROW_Width; NO canopy radius/height; layer 0 is "Park Trees" so confirm street-tree layer; trunk size is only a rough proxy, so imagery may add real value |
| 3D trees | `3D_Trees___Urban` (SceneServer) | Canopy volume/height | unknown | unknown | CC-BY-4.0 | Exists only | scene layer; extraction cost unknown |
| 3D buildings | `Boise_Buildings_3D/FeatureServer/0` | Enclosure / sky-view | citywide | unknown | CC-BY-4.0 | Schema seen | `BLDGHEIGHT` confirmed (+ ROOFFORM, BASEELEV, ROOFDIR); MultiPatch 3D (heavier to process); good for enclosure |
| Parcels | `Ada_County_Parcels/FeatureServer` | Setback / frontage | countywide | periodic | open | Exists only | large layer; clip to bbox |
| Zoning / land use | `Boise_Zoning`, `FutureLandUse`, `MixedUseZones` | Frontage / land-use context | citywide | periodic | CC-BY-4.0 | Exists only | - |
| Business improvement district | `Downtown_Business_Improvement_District` | Active-frontage prior | downtown | periodic | CC-BY-4.0 | Exists only | single polygon, coarse |
| OSM | Overpass | Walk graph + sidewalks | full | live | ODbL | Yes | sidewalks dense (~2,900 ways); lighting tags useless (115 lamps / 30 `lit`) |
| Transit (GTFS) | Valley Regional Transit (Transitland `f-9rv2-valleyride`) | Active transit at night | metro | periodic | CC-BY-3.0 | Feed exists | limited night service is itself a signal |
| POIs / businesses | Foursquare OS Places (Apache-2.0); OSM POIs | Open-business density | global / full | monthly (FSQ) | Apache-2.0 / ODbL | Not pulled | FSQ hours coverage to confirm |
| Crime incidents | `BPD_Crimes_Public/FeatureServer` | Validation / context only | citywide | periodic | open | Exists only | reporting + policing bias; NIBRS classes; never an input |
| Calls for service | `BPD_CallsForService/FeatureServer` | Validation / context only | citywide | periodic | open | Exists only | dispatch-driven; never an input |
| Streetlight conduit | `Streetlight_Conduit`, `PowerCabinets` | Lighting-infra context | citywide | unknown | CC-BY-4.0 | Exists only | secondary |
| Paths / parks | `ExistingPathways`, `BoiseParks`, trails | Isolation / off-street | citywide | periodic | CC-BY-4.0 | Exists only | "poorly visible park" flag |

## Key verdicts

1. The linchpin (streetlight inventory) exists, is public, authoritative, complete, and
   richly attributed. The lighting model's go-condition is met.
2. Boise is data-rich enough that structured proxies exist for "perception" features
   (occlusion via trees, enclosure via 3D buildings, frontage via zoning/parcels). This
   shifts the AI layer toward marginal value over a strong baseline.
3. Crime and demographics are validation/context only. Recorded here so the rule is
   explicit in the data layer, not just the model.

## Gaps found

- Outage data: no open queryable layer. The streetlight service has a single "Street
  Lights" layer, and outage reports live behind a web app, not a downloadable feed. The
  recent-outage freshness signal is therefore at risk and would need a city data request.
  Secondary, not core.
- Street trees: layer 0 is "Park Trees"; the street-tree layer still needs confirming,
  and there is no canopy-radius field.

## Still to verify (next structured pass)

- Confirm the street-tree layer (vs park trees) and whether trunk diameter is enough.
- 3D buildings: extract footprint + height efficiently from the MultiPatch layer.
- Foursquare OS Places hours coverage for the bbox.
- GTFS night-service span (last departures by stop).

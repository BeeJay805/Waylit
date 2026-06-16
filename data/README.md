# data/

Working data lives here. Everything in this folder is git-ignored except this README
(raw and derived data should not be committed).

Convention:

- `data/raw/`          untouched downloads (streetlight JSON, OSM extract, GTFS zip, imagery)
- `data/interim/`      cleaned and reprojected intermediates
- `data/processed/`    the final per-segment feature table + dataset release
- `data/ground_truth/` night-audit measurements (lux, ordinal ratings, photos)

Re-create everything from the scripts in `scripts/` plus `config/area.yaml`. Nothing
here is a source of truth; the source of truth is the public APIs plus the scripts.

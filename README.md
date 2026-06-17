# Waylit

An open, auditable, environment-based map of how comfortable a street is to walk at
night, for downtown Boise, Idaho. Waylit estimates physical and situational factors
(lighting, sidewalks, enclosure, active frontage, transit, isolation) and feeds them
into a transparent router that can suggest a calmer route and explain why.

Waylit does not claim any route is objectively "safe." It communicates uncertainty,
tracks provenance, and never uses crime data or demographics as model inputs.

## How it works

See [docs/architecture.md](docs/architecture.md) for the plain-language version. In
short: four layers (public data, an AI perception and inference layer, human night-audit
ground truth with active learning, and a transparent scoring and router), with the AI
layer as the research core.

## Status: Phase 0 feasibility sprint

We are validating data and method before the full build. See
[docs/phase0/README.md](docs/phase0/README.md) for the checklist and status.

Headline findings so far:

- Streetlight inventory is photometric and complete: 13,965 lamps citywide, ~3,091
  downtown (~2,860 active), with per-lamp wattage, height, fixture, lamp type, color
  temperature, and owner, ~98-100% complete across all owners including Idaho Power.
- That makes the structured baseline strong, so the research question is the marginal
  value of imagery plus adaptation on top of it.
- OSM has dense sidewalks but near-useless lighting tags, so lighting comes from the
  city inventory.
- One blocker: the imagery layer waits on a Mapillary access token.

## Constraints

Solo student, limited budget, laptop-class compute (pretrained inference plus small
heads plus light adaptation; no from-scratch training; cheap cloud GPU only when
justified). Public and open data preferred. One city first.

## Repo layout

```
config/       area + source configuration (area.yaml)
data/         raw + derived data (git-ignored; see data/README.md)
docs/         architecture + Phase 0 deliverables
scripts/      reproducible data pulls and probes
src/waylit/   pipeline code
```

## Reproduce the data checks

```
pip install -r requirements.txt
python scripts/probe_structured_sources.py
```

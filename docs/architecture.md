# How Waylit works (plain version)

Waylit turns public data about streets into a per-street-segment estimate of "how
comfortable is this to walk at night," and uses it to suggest a calmer route with
reasons. It is built in four layers.

## 1. Public structured data

Free, already-published facts about the city: streetlight locations (with wattage,
height, fixture type), the road and sidewalk network (OpenStreetMap), bus stops and
schedules (GTFS), businesses, land use, 3D building shapes, and street trees.

## 2. AI perception and inference (the research core)

Some things that matter at night are not in any dataset: is a tree blocking a lamp, how
walled-in or open a street feels, whether storefronts face the sidewalk. We estimate
these from daytime street-level photos (Mapillary) using vision models, then combine
them with the structured data.

One honesty rule is baked in:

- "Lighting potential" (how much light the street is built to have) can be estimated
  from daytime photos plus the lamp inventory.
- "Actual night brightness" can only be known by measuring at night (a lux meter, night
  photos), because a lamp can be burnt out or a tree can have grown.

We estimate potential, then learn how potential relates to real measured brightness. We
never claim a daytime photo tells us a street is lit right now.

## 3. Human ground truth and active learning

A person walks a sample of streets at night and records how bright they are. These
measurements are the answer key. "Active learning" means the model points us at the
streets it is least sure about, so limited night-walking effort is spent where it helps
most.

## 4. Deterministic scoring and routing

A simple, transparent formula combines the features into a comfort score with adjustable
weights, then a router compares the fastest route with a calmer route and explains the
trade-off. The score is reproducible and auditable; no black box decides it.

## The experiment: does AI actually help?

We build four versions and compare them fairly on the same night measurements:

- A: structured data only (no photos)
- B1: photos through a frozen pretrained vision model plus a small predictor
- B2: same, but we lightly adapt (fine-tune) the vision model on our own labels
- C: combine adapted photo features + structured data + the night measurements

We measure whether B2 and C beat A and B1 on accuracy, on calibrated uncertainty (do the
confidence ranges hold up), and on streets where structured data is missing.

## Hard rules

- Crime data and demographics are never model inputs (only used to check our work).
- We do not generate fake night images; we measure real ones.
- We never label a place "safe" or "unsafe."

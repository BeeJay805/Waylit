# Phase 0: night ground-truth pilot protocol (locked)

Purpose: collect a small, consistent set of night lighting measurements to (1) measure
how repeatable the labels are and how long each segment takes, and (2) seed the model
spike. Target: 30-50 segments. This protocol is locked so every measurement is taken the
same way.

## Unit: the segment

A segment is one block-face: a stretch of one side of a street between two intersections.
Identify it by the OSM edge id (preferred) or a generated id, plus the two cross-street
names.

## Which segments

Pick 30-50 to span the variety, not by random alone:

- a mix of fixture types (Historical, Cobrahead, Bollard) and owners,
- a mix of expected bright / dim / dark,
- some with heavy tree cover, some open,
- spatially spread across downtown, not one block.

Include 8-10 that you will measure twice (a second pass or a second night) to estimate
consistency.

## When

- Start at least 60-90 minutes after sunset (full dark).
- Skip rain or snow. Record weather and rough temperature.
- Record the clock time per segment (lamps and open businesses change through the night).

## What to record, per segment

1. Ordinal brightness, 1-4 (the primary label):
   - 1 Dark: you cannot clearly see a person's face or the ground a few metres ahead.
   - 2 Dim: you can see the ground, but shadows dominate and detail is lost.
   - 3 Adequate: you can see clearly, with some darker patches.
   - 4 Bright: evenly well-lit the length of the segment.
2. Lux reading (where practical): phone lux app or a ~$30 lux meter. Hold the sensor
   horizontal at waist height, facing up, at mid-segment. Record one reading at the
   darkest spot (between lamps) and note if you are directly under a lamp.
3. Two photos under locked camera settings: same phone, manual exposure if possible (fix
   ISO and shutter; if not, note auto). Frame one down the sidewalk in the walking
   direction and one across the street. Locked settings make photos comparable.
4. Optional comfort note (1-5), kept separate from the physical readings.

## Metadata (every row)

CSV header (saved to `data/ground_truth/`, git-ignored):

```
segment_id,lat,lon,datetime,weather,temp_f,observer,instrument,lamp_context,ordinal,lux,comfort,photo1,photo2,notes
```

## Consistency check

Re-measure the 8-10 repeat segments and report:

- ordinal agreement (exact match, and within plus or minus 1),
- lux agreement (ratio of the two readings).

This tells us how noisy a single human label is before we trust the model against it.

## Safety (named plainly)

This project studies the risk of walking alone at night, so do not pilot it in a way
that puts you at risk. Recommended: scout segments in daylight first, bring a companion
for the night pass, share your route and check-in times, and start with well-trafficked
blocks. The auditor's safety comes first; skip any segment that feels unsafe and record
it as skipped.

## Time budget

We do not know the per-segment time yet; measuring it is a goal of this pilot. Hypothesis
to test: about 3-5 minutes per segment plus walking, so 30-50 segments across one or two
nights in 2-3 hours.

## Output of the pilot

A small CSV, the repeat-measure consistency numbers, and the actual time per segment.
These feed the go/no-go memo and the first model spike.

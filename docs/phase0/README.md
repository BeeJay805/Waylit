# Phase 0: feasibility sprint

Goal: decide whether Boise is the right city and whether the AI layer is worth building,
using small, cheap checks before committing to the full build. Novelty stays a
hypothesis until this is done.

## Deliverables and status

| # | Deliverable | What it answers | Status |
|---|---|---|---|
| 1 | [structured-data-availability.md](structured-data-availability.md) | Does the public data exist and is it usable? | Done (verified live) |
| 2 | [prior-art.md](prior-art.md) | Has someone already done this? What do we add? | Drafted |
| 3 | [ground-truth-protocol.md](ground-truth-protocol.md) | How do we measure night lighting consistently? | Drafted |
| 4 | [coverage-report.md](coverage-report.md) | Is there usable daytime imagery in Boise? | Done: 78% of cells covered (caveats) |
| 5 | model spike | Does AI beat the structured baseline on a tiny sample? | Pending pilot labels |
| 6 | [go-no-go.md](go-no-go.md) | Go, adjust, or switch city? | Open (collecting evidence) |

## How to re-run the data checks

```
python scripts/probe_structured_sources.py
```

Prints the counts and schemas behind the availability matrix.

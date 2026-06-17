"""Tiny read-only helpers for ArcGIS REST Feature Services.

Standard library only (no third-party deps) so the data checks run anywhere Python
3.9+ is installed. Used by scripts/probe_structured_sources.py.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

TIMEOUT = 45


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.load(resp)


def _envelope(bbox) -> dict:
    """bbox = (min_lon, min_lat, max_lon, max_lat) in WGS84, or None."""
    if not bbox:
        return {}
    return {
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def query(layer_url: str, **params) -> dict:
    """Run a /query against a FeatureServer layer URL (.../FeatureServer/0)."""
    params.setdefault("f", "json")
    return _get(layer_url + "/query?" + urllib.parse.urlencode(params))


def fields(layer_url: str) -> list[dict]:
    """Return the field definitions of a layer."""
    return _get(layer_url + "?f=json").get("fields", [])


def count(layer_url: str, where: str = "1=1", bbox=None) -> int:
    params = {"where": where, "returnCountOnly": "true", **_envelope(bbox)}
    return query(layer_url, **params).get("count", 0)


def group_count(layer_url: str, group_field: str, where: str = "1=1", bbox=None) -> dict:
    """Count features grouped by a field value (e.g. Pole_Owner -> count)."""
    stats = [{"statisticType": "count", "onStatisticField": "OBJECTID",
              "outStatisticFieldName": "n"}]
    params = {"where": where, "groupByFieldsForStatistics": group_field,
              "outStatistics": json.dumps(stats), **_envelope(bbox)}
    out = {}
    for feat in query(layer_url, **params).get("features", []):
        attrs = feat["attributes"]
        out[attrs.get(group_field)] = attrs.get("n")
    return out

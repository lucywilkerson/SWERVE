"""HAPI metadata for source/site/type/class/variant identifiers."""

import json
import math
import numpy as np

from .swerve_to_numpy import _root, _series


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def info(id, *, data_root=None):
    """Given a HAPI id, return its info object across available cache partitions."""
    parts = id.split("/")
    if len(parts) != 5 or parts[-1] not in ("original", "modified"):
        raise ValueError("Expected source/site/type/class/{original,modified}")
    source, site, quantity, kind, variant = parts
    start, stop, signature, partitions = None, None, None, []
    for path, frame, details in _series(source, site, quantity, kind, variant == "modified", data_root):
        current = (tuple(frame.columns), details["unit"])
        if signature is not None and current != signature:
            raise ValueError(f"Inconsistent parameters or units across caches: {path}")
        signature = current
        first, last = frame.index.min(), frame.index.max()
        start = first if start is None else min(start, first)
        stop = last if stop is None else max(stop, last)
        relative = path.relative_to(_root(data_root)).as_posix()
        partitions.append({"sourceFile": relative, "startDate": first.isoformat(),
                           "stopDate": last.isoformat(), "cachedRecords": len(frame), **details})
    if signature is None:
        raise ValueError(f"No cached records for {id}")
    labels, unit = signature
    start_text, stop_text = (time.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for time in (start, stop))
    citation_file = _root(data_root) / "data_original" / source.lower() / "info.json"
    citations = json.loads(citation_file.read_text()) if citation_file.is_file() else None
    content = {"source": source, "siteID": site, "dataType": quantity, "dataClass": kind,
               "variant": variant, "partitions": partitions,
               "citationMetadataStatus": "present" if citations is not None else "missing",
               "citations": citations}
    result = {
        "startDate": start_text, "stopDate": stop_text,
        "description": f"SWERVE cached {variant} {quantity} ({kind}) from {source} at {site}, {start_text} to {stop_text}.",
        "parameters": [{"name": "Time", "type": "isotime", "length": 27, "units": "UTC", "fill": None}]
                      + [{"name": label, "type": "double", "units": unit, "fill": "-1e31"} for label in labels],
        "additionalMetadata": [{"name": "SWERVE", "content": _json_value(content),
                                "aboutURL": "https://github.com/lucywilkerson/SWERVE-data"}],
    }
    if len(partitions) == 1 and len(frame) > 1:
        steps = frame.index.to_series().diff().dropna().dt.total_seconds()
        if steps.nunique() == 1 and steps.iloc[0] > 0:
            seconds = f"{steps.iloc[0]:.6f}".rstrip("0").rstrip(".")
            result["cadence"] = f"PT{seconds}S"
    return result

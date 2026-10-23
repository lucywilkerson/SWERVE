"""Build HAPI catalog files from a SWERVE-data checkout."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .inventory import discover
from .layout import FILL, ISO, KINDS, QUANTITY_NAMES, SOURCES, UNITS, Dataset
from .readers import read

READ_ERRORS = (OSError, ValueError, KeyError, OverflowError)
SWERVE_DATA_URL = "https://github.com/lucywilkerson/SWERVE-data"
PROVIDERS = {
    "NERC": "the NERC GMD data portal",
    "TVA": "the Tennessee Valley Authority",
    "GMU": "the George Mason University C-SWIM simulation",
    "SWMF": "the Space Weather Modeling Framework",
    "OpenGGCM": "OpenGGCM",
    "MAGE": "the MAGE model",
}

def catalog():
    """Return list of dataset IDs of form "data_source/site_id/data_type/data_class/{original,modified}"."""
    datasets = discover()
    ids = []
    for dataset in datasets:
        for variant in ("original", "modified"):
            ids.append(f"{dataset.source}/{dataset.site}/{dataset.quantity}/{dataset.kind}/{variant}")
    return ids

def importance(entry: dict) -> tuple:
    """Catalog order: measured before calculated, GIC before B, source order, newest storm first, then site."""
    content = entry["info"]["additionalMetadata"][0]["content"]
    return (
        KINDS.index(content["dataClass"]),
        list(UNITS).index(content["dataType"]),
        SOURCES.index(content["source"]) if content["source"] in SOURCES else len(SOURCES),
        content["source"].lower(),
        tuple(-int(part) for part in content["event"].split("-")),
        entry["id"],
    )


def describe(dataset: Dataset, data: pd.DataFrame, cadence: str | None, max_gap: float | None) -> str:
    start, stop = data.index[0].strftime(ISO), data.index[-1].strftime(ISO)
    if cadence is not None:
        spacing = f"at a uniform {cadence} cadence"
    elif max_gap is not None:
        spacing = f"with irregular spacing (largest gap {max_gap:g} s)"
    else:
        spacing = "as a single record"
    if dataset.product == "processed":
        provenance = "Values are SWERVE's stored modified arrays; see additionalMetadata for the recorded modification and site quality flags. No further resampling or filtering is applied by this reader."
    else:
        provenance = (f"Values are as provided by {PROVIDERS[dataset.source]}, at the original cadence and baseline, "
                      "without resampling, filtering, or component sums.")
    return (
        f"{dataset.kind.capitalize()} {QUANTITY_NAMES[dataset.quantity]} at {dataset.source} site {dataset.site} "
        f"during the {dataset.event} geomagnetic storm: {len(data)} records from {start} to {stop} {spacing}. "
        f"{provenance} Read from SWERVE-data {', '.join(dataset.files)}."
    )


def metadata(dataset: Dataset, data: pd.DataFrame, details: dict, timezone: str) -> dict:
    """HAPI ``info`` response for one dataset; non-HAPI metadata goes in ``additionalMetadata``."""
    parameters = [{"name": "Time", "type": "isotime", "length": 27, "units": "UTC", "fill": None}]
    parameters += [{"name": name, "type": "double", "units": dataset.unit, "fill": FILL} for name in data.columns]
    content = {
        "source": dataset.source,
        "siteId": dataset.site,
        "dataType": dataset.quantity,
        "dataClass": dataset.kind,
        "dataProduct": dataset.product,
        "event": dataset.event,
        "sourceTimeZone": timezone,
        "records": len(data),
        "fillCounts": {name: int(count) for name, count in data.isna().sum().items()},
        **dataset.details,
        **details,
        "sourceFiles": list(dataset.files),
    }
    if dataset.location is not None:
        content["location"] = dict(zip(("latitude", "longitude"), dataset.location))
    cadence, max_gap = None, None
    steps = data.index.to_series().diff().dropna().dt.total_seconds()
    if not steps.empty:
        max_gap = float(steps.max())
        content["maxGapSeconds"] = max_gap
        if steps.nunique() == 1:
            seconds = f"{steps.iloc[0]:.6f}".rstrip("0").rstrip(".")
            cadence = f"PT{seconds}S"
    info = {
        "startDate": data.index[0].strftime(ISO),
        "stopDate": data.index[-1].strftime(ISO),
        "description": describe(dataset, data, cadence, max_gap),
        "parameters": parameters,
        "additionalMetadata": [{"name": "SWERVE", "content": content, "aboutURL": SWERVE_DATA_URL}],
    }
    if cadence is not None:
        info["cadence"] = cadence
    return {"id": dataset.id, "title": dataset.title, "info": info}


def build(root: Path, timezone: str, output: Path, only_events=None, only_sources=None, workers=4, log=None, product="original") -> int:
    """Write catalog.json, catalog-all.json, and catalog-report.json. Returns 0 if any dataset was included."""
    log = sys.stderr if log is None else log
    datasets, notes = discover(root, only_events, only_sources, product=product)

    def inspect(dataset):
        try:
            data, details = read(dataset, root, timezone)
            return metadata(dataset, data, details, timezone), None
        except READ_ERRORS as error:
            return None, {"id": dataset.id, "reason": str(error)}

    catalog, excluded = [], []
    # Only metadata is retained; at most `workers` source frames are in memory.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (entry, error) in enumerate(pool.map(inspect, datasets), 1):
            if entry is not None:
                catalog.append(entry)
            else:
                excluded.append(error)
            if number % 100 == 0:
                print(f"Checked {number}/{len(datasets)} datasets", file=log, flush=True)
    catalog.sort(key=importance)

    per_event = {}
    for entry in catalog:
        per_event.setdefault(entry["info"]["additionalMetadata"][0]["content"]["event"], {"included": 0, "excluded": 0})["included"] += 1
    for error in excluded:
        per_event.setdefault(error["id"].rsplit("/", 1)[1], {"included": 0, "excluded": 0})["excluded"] += 1
    report = {
        "dataRoot": str(root),
        "sourceTimeZone": timezone,
        "dataProduct": product,
        "candidates": len(datasets),
        "included": len(catalog),
        "events": dict(sorted(per_event.items())),
        "excluded": excluded,
        "notes": notes,
    }
    outputs = {
        "catalog.json": [{"id": entry["id"], "title": entry["title"]} for entry in catalog],
        "catalog-all.json": catalog,
    }
    report["files"] = {name: hashlib.sha256(encode(value).encode()).hexdigest() for name, value in outputs.items()}
    outputs["catalog-report.json"] = report
    write(output, outputs)
    print(f"Wrote {len(catalog)} datasets; excluded {len(excluded)}; {len(notes)} notes. Report: {output / 'catalog-report.json'}", file=log)
    return 0 if catalog else 1


def encode(value) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def write(output: Path, outputs: dict) -> None:
    """Publish the files in two phases.

    Every file is serialised and written to a ``.json.tmp`` neighbour first; only
    then is each one renamed into place. Each rename is atomic, so a reader never
    sees a partial file, but a crash between renames can leave files from two
    builds side by side. ``catalog-report.json`` therefore records the SHA-256 of
    the two catalog files it was built with, and is renamed last.
    """
    encoded = {name: encode(value) for name, value in outputs.items()}
    output.mkdir(parents=True, exist_ok=True)
    temporaries = {}
    for name, content in encoded.items():
        temporary = (output / name).with_suffix(".json.tmp")
        temporary.write_text(content)
        temporaries[name] = temporary
    for name, temporary in temporaries.items():
        temporary.replace(output / name)

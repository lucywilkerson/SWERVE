"""Read SWERVE caches as [UTC datetime, value] arrays."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ..util.source_catalog import (ANNUAL_SOURCES, DATE_PARTITION_PATTERN,
                                   YEAR_PARTITION_PATTERN, source_kind)


def _root(data_root=None):
    # The same sibling-directory layout used by swerve.config(dirs_only=True).
    return Path(data_root) if data_root is not None else Path(__file__).resolve().parents[3] / "SWERVE-data"


def _cache_files(data_source, site_id, data_type, data_class, modified, data_root=None):
    source_kind(data_source)  # Validate against SWERVE's own source registry.
    if data_type not in ("GIC", "B", "DMM") or data_class not in ("measured", "calculated"):
        raise ValueError("Unsupported SWERVE data type or class")
    site = str(site_id).lower().replace(" ", "")
    if not site or site in (".", "..") or "/" in site or "\\" in site:
        raise ValueError("Invalid site ID")
    directory = _root(data_root) / "data_processed"
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed SWERVE data not found: {directory}")
    filename = f"{data_type}_{data_class}_{data_source}.pkl"
    if data_source in ANNUAL_SOURCES and not modified:
        annual = directory / "non-event"
        paths = (annual / year.name / site / filename for year in sorted(annual.iterdir())
                 if YEAR_PARTITION_PATTERN.fullmatch(year.name)) if annual.is_dir() else ()
    else:
        paths = (event / "sites" / site / "data" / filename for event in sorted(directory.iterdir())
                 if DATE_PARTITION_PATTERN.fullmatch(event.name))
    return [path for path in paths if path.is_file()]


def _read_series(file_path, data_source, data_type, data_class, modified):
    # SWERVE's own caches are trusted local pickle files, as in site_read.py.
    with Path(file_path).open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid SWERVE cache: {file_path}")
    entry = cache.get(data_source, cache)
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid provider entry in {file_path}")
    variant = "modified" if modified else "original"
    if variant in entry:
        series = entry[variant]
    elif not modified and "time" in entry and "data" in entry:
        series = entry  # Annual non-event caches store the original series directly.
    else:
        return None
    if not isinstance(series, dict) or series.get("error"):
        raise ValueError(f"Invalid {variant} series in {file_path}")
    if any(key not in series for key in ("time", "data", "labels", "unit")):
        raise ValueError(f"Incomplete {variant} series in {file_path}")
    time = pd.DatetimeIndex(pd.to_datetime(series["time"], utc=True))
    values = np.asarray(series["data"], dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if time.hasnans or values.ndim != 2 or len(time) != len(values) or not values.shape[1]:
        raise ValueError(f"Invalid time/data dimensions in {file_path}")
    if time.nanosecond.any():
        raise ValueError("Sub-microsecond timestamps cannot be returned as Python datetimes")
    labels = list(series["labels"])
    note = None
    # site_read appends B_H to the shared labels list after reading original B.
    if not modified and data_type == "B" and values.shape[1] == 3 and len(labels) == 4 and labels[-1] == "B_H":
        labels = labels[:3]
        note = "Removed the upstream-appended B_H label from original three-column data."
    if (len(labels) != values.shape[1] or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label or label == "Time" for label in labels)):
        raise ValueError(f"Parameter labels do not match the data in {file_path}")
    if not isinstance(series["unit"], str) or not series["unit"]:
        raise ValueError(f"Missing or invalid units in {file_path}")
    values = np.where(np.isfinite(values), values, np.nan)
    if np.any(values == -1e31):
        raise ValueError(f"A value collides with the HAPI fill in {file_path}")
    details = {"unit": series["unit"], "modification": series.get("modification")}
    site = Path(file_path).parent.parent.name if Path(file_path).parent.name == "data" else Path(file_path).parent.name
    for name, value in entry.items():
        if isinstance(value, dict) and str(name).lower().replace(" ", "") == site:
            details["siteMetadata"] = value
    if note:
        details["labelNote"] = note
    return pd.DataFrame(values, index=time, columns=labels), details


def _series(data_source, site_id, data_type, data_class, modified, data_root=None):
    for path in _cache_files(data_source, site_id, data_type, data_class, modified, data_root):
        series = _read_series(path, data_source, data_type, data_class, modified)
        if series is not None and not series[0].empty:
            yield path, *series


def swerve_to_numpy(data_source, site_id, data_type, data_class, modified, start, stop, *, data_root=None):
    """Return an (n, 2) object array of [UTC datetime, scalar or component array].

    Naive request times and cache times mean UTC. The interval is [start, stop).
    Missing files are skipped; original data never replace missing modified data.
    Identical overlapping records are deduplicated; conflicting records raise.
    """
    if not isinstance(modified, (bool, np.bool_)):
        raise TypeError("modified must be True or False")
    start, stop = [pd.to_datetime(value, utc=True) for value in (start, stop)]
    if pd.isna(start) or pd.isna(stop) or start >= stop:
        raise ValueError("Use valid start and stop times with start < stop")
    frames, signature = [], None
    for path, frame, details in _series(data_source, site_id, data_type, data_class, modified, data_root):
        frame = frame.loc[(frame.index >= start) & (frame.index < stop)]
        if frame.empty:
            continue
        current = (tuple(frame.columns), details["unit"])
        if signature is not None and current != signature:
            raise ValueError(f"Inconsistent parameters or units across caches: {path}")
        signature = current
        frames.append(frame)
    if not frames:
        return np.empty((0, 2), dtype=object)
    frame = pd.concat(frames).sort_index(kind="stable")
    duplicates = frame.index.duplicated(keep=False)
    if duplicates.any():
        conflicts = frame.loc[duplicates].groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"Conflicting overlapping SWERVE values at {conflicts[conflicts].index[0]}")
        frame = frame.loc[~frame.index.duplicated()]
    result = np.empty((len(frame), 2), dtype=object)
    for index, (timestamp, values) in enumerate(zip(frame.index, frame.to_numpy())):
        result[index, 0] = timestamp.to_pydatetime()
        result[index, 1] = values[0] if len(values) == 1 else values.copy()
    return result

"""List the HAPI IDs available in SWERVE's native cache files."""

import json
from pathlib import Path
from pickle import UnpicklingError
import sys

from .swerve_to_numpy import _root, _read_series
from ..util.source_catalog import ANNUAL_SOURCES, DATE_PARTITION_PATTERN, YEAR_PARTITION_PATTERN, source_kind


def catalog(*, data_root=None, output=Path(__file__).resolve().parents[2] / "catalog.json"):
    """Return IDs: source/site/type/class/{original,modified}; write them as JSON.

    Each ID spans its available event or annual partitions. Only variants with
    cached records are listed. Pass output=None to return without writing.
    """
    directory = _root(data_root) / "data_processed"
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed SWERVE data not found: {directory}")
    ids = set()
    paths = sorted(directory.glob("*/sites/*/data/*.pkl")) + sorted(directory.glob("non-event/*/*/*.pkl"))
    for path in paths:
        if path.name == "_all.pkl":
            continue
        parts = path.stem.split("_", 2)
        if len(parts) != 3:
            continue
        data_type, data_class, source = parts
        relative = path.relative_to(directory).parts
        annual = relative[0] == "non-event"
        partition = relative[1] if annual else relative[0]
        pattern = YEAR_PARTITION_PATTERN if annual else DATE_PARTITION_PATTERN
        if not pattern.fullmatch(partition):
            continue
        site = path.parent.name if annual else path.parent.parent.name
        try:
            source_kind(source)
        except ValueError as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
            continue
        if data_type not in ("GIC", "B", "DMM") or data_class not in ("measured", "calculated"):
            continue
        for variant in ("original", "modified"):
            if annual != (source in ANNUAL_SOURCES and variant == "original"):
                continue
            try:
                item = _read_series(path, source, data_type, data_class, variant == "modified")
                if item is not None and not item[0].empty:
                    ids.add(f"{source}/{site}/{data_type}/{data_class}/{variant}")
            except (OSError, ValueError, TypeError, EOFError, UnpicklingError) as error:
                print(f"Skipping {variant} in {path}: {error}", file=sys.stderr)
    def importance(identifier):
        source, site, quantity, kind, variant = identifier.split("/")
        sources = ("NERC", "TVA", "Bor", "GMU", "SWMF", "OpenGGCM", "MAGE")
        return (kind != "measured", {"GIC": 0, "B": 1, "DMM": 2}.get(quantity, 3),
                sources.index(source) if source in sources else len(sources), source.lower(), site, variant != "original")
    ids = sorted(ids, key=importance)
    if output is not None:
        Path(output).write_text(json.dumps(ids, indent=2) + "\n")
    return ids


def catalog_all(*, data_root=None, output=Path(__file__).resolve().parents[2] / "catalog-all.json"):
    """Return IDs with metadata; write JSON unless output=None.

    Build when the caches change. Deployment can load the JSON once and reuse
    its metadata instead of rebuilding it for each request.
    """
    from .info import info

    ids = catalog(data_root=data_root, output=None)
    entries = [{"id": identifier, "info": info(identifier, data_root=data_root)} for identifier in ids]
    if output is not None:
        Path(output).write_text(json.dumps(entries, indent=2, allow_nan=False) + "\n")
    return entries

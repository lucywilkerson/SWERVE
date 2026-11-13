import json
import os
from pathlib import Path
import re

def load_source_catalog(original_dir=None):
  """Read source names and kinds from data_original/<data_source>/info.json."""
  directory = (Path(original_dir) if original_dir is not None
               else Path(__file__).resolve().parents[3] / 'SWERVE-data' / 'data_original')
  if not directory.is_dir():
    raise FileNotFoundError(f"Source metadata directory not found: {directory}")

  kinds = {'event_partitioned': 'event', 'annual': 'annual', 'continuous': 'continuous'}
  sources = {}
  for info_file in sorted(directory.glob('*/info.json')):
    with info_file.open(encoding='utf-8') as handle:
      info = json.load(handle)
    if not isinstance(info, dict):
      raise ValueError(f"Invalid source metadata in {info_file}: expected an object.")
    name = info.get('source_name')
    source_type = info.get('source_type')
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f"Missing or invalid source_name in {info_file}.")
    if not isinstance(source_type, str) or source_type not in kinds:
      raise ValueError(f"Invalid source_type {source_type!r} in {info_file}.")
    if name in sources:
      raise ValueError(f"Duplicate source_name '{name}' in {info_file}.")
    sources[name] = kinds[source_type]
  if not sources:
    raise ValueError(f"No source info.json files found in {directory}.")
  return sources


SOURCE_KINDS = load_source_catalog()
EVENT_PARTITIONED_SOURCES = frozenset(name for name, kind in SOURCE_KINDS.items() if kind == 'event')
ANNUAL_SOURCES = frozenset(name for name, kind in SOURCE_KINDS.items() if kind == 'annual')
CONTINUOUS_SOURCES = frozenset(name for name, kind in SOURCE_KINDS.items() if kind == 'continuous')

DATE_PARTITION_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
YEAR_PARTITION_PATTERN = re.compile(r'^\d{4}$')


def source_kind(data_source, original_dir=None):
  sources = SOURCE_KINDS if original_dir is None else load_source_catalog(original_dir)
  if data_source not in sources:
    raise ValueError(f"Unknown data source '{data_source}'. Add its info.json to data_original.")
  return sources[data_source]


def source_partitions(original_dir, data_source):
  """Return the source's native local data partitions."""
  kind = source_kind(data_source, original_dir)
  if data_source == 'INTERMAG':
    return []
  if kind == 'continuous':
    return ['unpartitioned']

  source_dir = os.path.join(original_dir, data_source.lower())
  if not os.path.isdir(source_dir):
    raise FileNotFoundError(f"Data for data source '{data_source}' not found at {source_dir}.")

  pattern = DATE_PARTITION_PATTERN if kind == 'event' else YEAR_PARTITION_PATTERN
  return sorted(
    entry for entry in os.listdir(source_dir)
    if pattern.match(entry) and os.path.isdir(os.path.join(source_dir, entry))
  )
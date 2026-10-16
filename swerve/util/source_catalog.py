import os
import re


EVENT_PARTITIONED_SOURCES = frozenset({
  'AlvesRibeiro',
  'Bailey',
  'Blake',
  'Caraballo2025',
  'Espinosa',
  'Bor',
  'Marsal2021',
  'Marsal2025',
  'Nahayo',
  'NERC',
  'OpenGGCM',
  'Parry2024',
  'Parry2025',
  'SWMF',
  'MAGE',
  'TVA',
  'Watari',
  'Zhang2020',
})

ANNUAL_SOURCES = frozenset({
  'Caraballo2023',
  'Hubert',
  'Hughes',
})

CONTINUOUS_SOURCES = frozenset({
  'INTERMAG',
  'TEST',
})

DATE_PARTITION_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
YEAR_PARTITION_PATTERN = re.compile(r'^\d{4}$')


def source_kind(data_source):
  if data_source in EVENT_PARTITIONED_SOURCES:
    return 'event'
  if data_source in ANNUAL_SOURCES:
    return 'annual'
  if data_source in CONTINUOUS_SOURCES:
    return 'continuous'
  raise ValueError(f"Unknown data source '{data_source}'. Add it to swerve.util.source_catalog.")


def source_partitions(original_dir, data_source):
  """Return the source's native local data partitions."""
  kind = source_kind(data_source)
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
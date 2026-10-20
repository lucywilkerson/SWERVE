def write_event_dict(conf, file_path, data_dir):
  """Write events_dict.json for a run configuration."""
  import datetime
  import json
  import os
  import re
  from datetime import timedelta

  from ..storm_time import storm_time
  from .source_catalog import ANNUAL_SOURCES, EVENT_PARTITIONED_SOURCES, source_partitions

  original_dir = os.path.join(data_dir, 'data_original')
  events = conf.get('event', None)
  event_types = {}
  start_time = conf.get('start_time')
  stop_time = conf.get('stop_time')
  data_sources = conf.get('data_source') or []
  if isinstance(data_sources, str):
    data_sources = [data_sources]

  def as_datetime(value):
    if isinstance(value, datetime.datetime):
      return value
    if isinstance(value, datetime.date):
      return datetime.datetime.combine(value, datetime.time.min)
    if isinstance(value, str):
      return datetime.datetime.fromisoformat(value)
    return value

  if data_sources == ['INTERMAG'] and not conf.get('event') and start_time is None:
    raise ValueError("For INTERMAG data source, either an event or start_time/stop_time must be specified.")

  if events is None and start_time is not None:
    events = [as_datetime(start_time).strftime('%Y-%m-%d')]
  elif events is None or events == []:
    raise ValueError("No event or start_time/stop_time specified in configuration file.")

  if events == 'all' or events == ['all']:
    event_sources = [source for source in data_sources if source in EVENT_PARTITIONED_SOURCES]
    annual_sources = [source for source in data_sources if source in ANNUAL_SOURCES]
    if not event_sources and not annual_sources:
      raise ValueError(
        "event='all' requires at least one event-partitioned or annual data source."
      )

    for data_source in event_sources:
      for event in source_partitions(original_dir, data_source):
        event_types[event] = 'event'
    for data_source in annual_sources:
      for year in source_partitions(original_dir, data_source):
        event_types[year] = 'annual'
    if not event_types:
      raise ValueError(f"No event or annual partitions found for data sources {data_sources}.")
    events = sorted(event_types)

  elif isinstance(events, str):
    events = [events]

  events = [str(event) for event in events]
  for event in events:
    event_types.setdefault(event, 'event')

  event_dict = {
    event: {'event_type': event_types[event], 'data_limits': None, 'plot_limits': None}
    for event in events
  }

  for event in events:
    if event_dict[event]['event_type'] == 'annual':
      year = int(event)
      event_dict[event]['data_limits'] = [
        datetime.datetime(year, 1, 1),
        datetime.datetime(year + 1, 1, 1) - timedelta(seconds=1),
      ]
      event_dict[event]['plot_limits'] = event_dict[event]['data_limits'].copy()
      continue

    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', event):
      raise ValueError(
        f"Event '{event}' is not a storm date. Use YYYY-MM-DD, event='all', "
        "or set start_time and stop_time without an event for a custom interval."
      )

    if start_time is not None and stop_time is not None:
      event_dict[event]['data_limits'] = [as_datetime(start_time), as_datetime(stop_time)]

    else:
      event_dict[event]['data_limits'] = storm_time(event)

    event_dict[event]['plot_limits'] = [
      event_dict[event]['data_limits'][0] - timedelta(hours=2),
      event_dict[event]['data_limits'][1],
    ]

  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', conf.get('run_config_name', 'default')))
  os.makedirs(info_dir, exist_ok=True)
  with open(os.path.join(info_dir, 'events_dict.json'), 'w') as file_handle:
    json.dump(event_dict, file_handle, indent=4, default=str)

  return event_dict
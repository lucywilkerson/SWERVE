import numpy
import pandas


def _utc_timestamp(value):
  timestamp = pandas.Timestamp(value)
  if timestamp.tzinfo is None:
    return timestamp.tz_localize('UTC')
  return timestamp.tz_convert('UTC')


def read_intermag_info(start, stop, sid=None, logger=None):
  from hapiclient import hapi

  if logger is None:
    from swerve import config
    CONFIG = config()
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  # Read in info for info.csv from available INTERMAG datasets
  logger.info("      Reading HAPI INTERMAG site information")
  server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
  catalog = hapi(server)
  datasets = [
    entry['id'] for entry in catalog['catalog']
    if entry['id'].endswith('/best-avail/PT1M/xyzf')
  ]
  if sid is not None:
    dataset_prefix = f'{sid.lower()}/'
    datasets = [dataset for dataset in datasets if dataset.startswith(dataset_prefix)]
  # Reformat start/stop times
  event_start = _utc_timestamp(start)
  event_stop = _utc_timestamp(stop)
  source_sites = {'sites': [], 'lat': [], 'lon': [], 'datasets': []}
  for dataset in datasets:
    meta = hapi(server, dataset)
    dataset_start = _utc_timestamp(meta['startDate'])
    dataset_stop = _utc_timestamp(meta['stopDate'])
    if dataset_stop < event_start or dataset_start > event_stop:
      continue

    sid = dataset.split('/', 1)[0].upper()
    source_sites['sites'].append(sid)
    source_sites['lat'].append(meta['x_latitude'])
    source_sites['lon'].append(meta['x_longitude'])
    source_sites['datasets'].append(dataset)

  # Sort sites by latitude
  sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon'], source_sites['datasets']))
  if not sorted_sites:
    return source_sites
  source_sites['lat'], source_sites['sites'], source_sites['lon'], source_sites['datasets'] = zip(*sorted_sites)

  return source_sites


def read_intermag(start, stop, sid=None, logger=None):
  from hapiclient import hapi, hapitime2datetime

  if logger is None:
    from swerve import config
    CONFIG = config()
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  # Read INTERMAG data for given start/stop and site
  logger.info("      Using HAPI INTERMAG data")
  server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
  source_sites = read_intermag_info(start, stop, sid=sid, logger=logger)
  start = _utc_timestamp(start).isoformat().replace('+00:00', 'Z')
  stop = _utc_timestamp(stop).isoformat().replace('+00:00', 'Z')

  intermag_df = pandas.DataFrame()
  parameters = 'Field_Vector'
  for sid, dataset in zip(source_sites['sites'], source_sites['datasets']):
    data, meta = hapi(server, dataset, parameters, start, stop)

    dfs = []
    dfs.append(pandas.DataFrame({'Timestamp': hapitime2datetime(data['Time'])}))

    field_vector = data['Field_Vector']
    for i, component in enumerate(['X', 'Y', 'Z']):
      dfs.append(pandas.DataFrame({component: field_vector[:, i]}))
      fill = meta['parameters'][1]['fill']
      if fill is not None:
        fill = float(fill)
        dfs[-1][dfs[-1][component] == fill] = numpy.nan  # remove erroneous high values
      dfs[-1][component] = dfs[-1][component] - numpy.nanmedian(dfs[-1][component])  # remove median offset

    df = pandas.concat(dfs, axis=1)
    intermag_df = pandas.concat([intermag_df, df.assign(site_id=sid)], ignore_index=True)

  return intermag_df, source_sites

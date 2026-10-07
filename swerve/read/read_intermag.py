import numpy
import pandas


def read_intermag(start, stop, logger=None):
  from hapiclient import hapi, hapitime2datetime

  if logger is None:
    from swerve import config

    CONFIG = config()
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  logger.info("Using HAPI INTERMAG data")
  server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
  catalog = hapi(server)
  datasets = [
    entry['id'] for entry in catalog['catalog']
    if entry['id'].endswith('/best-avail/PT1M/xyzf')
  ]

  intermag_df = pandas.DataFrame()
  source_sites = {'sites': [], 'lat': [], 'lon': []}
  parameters = 'Field_Vector'
  for dataset in datasets:
    sid = dataset.split('/', 1)[0].upper()
    data, meta = hapi(server, dataset, parameters, start, stop)

    source_sites['sites'].append(sid)
    source_sites['lat'].append(meta['x_latitude'])
    source_sites['lon'].append(meta['x_longitude'])

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

  # Sort sites by latitude
  sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
  source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)

  return intermag_df, source_sites

import os
import numpy
import pickle

from .orig_readers import _site_read_orig
from .output_error import _output_error
from .. import _write_pkl

def site_read(sid, event, data_types=None, reparse=False, start=None, stop=None, add_errors=False, logger=None, debug=False):
  """Read data from one or more sites

  Usage:
    site_read(sid, event, data_types=None, reparse=False, logger=None):

  If `data_types` is None, read all data types (e.g, B, GIC) for the site.

  If 'add_errors' is True, add automated error checks to data.

  If `reparse` is True, reparse the data files even if cache file exists
  (use if data files or code in this script that reads them has changed).

  If `debug` is True, print processing details for computing resampled data.
  """
  from swerve import config, read_info_dict, resample

  CONFIG = config()

  if logger is None:
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  resample_kwargs = {}
  if debug:
    resample_kwargs = {'logger': logger, 'logger_indent': 6}

  sidx = sid.lower().replace(' ', '')
  site_all_file = '_all.pkl'
  out_dir = CONFIG['dirs']['processed']
  site_all_file = os.path.join(CONFIG['dirs']['data'], out_dir, event, 'sites', sidx, 'data', site_all_file)

  logger.info(f"Reading '{sid}' data for event '{event}'")

  if not reparse:
    if os.path.exists(site_all_file):
      logger.info(f"  Reading cached file with all data for site '{sid}': {site_all_file}")
      with open(site_all_file, 'rb') as f:
        data = pickle.load(f)
        return data

  if start is None:
    start = CONFIG['event'][event]['data_limits'][0]
  if stop is None:
    stop = CONFIG['event'][event]['data_limits'][1]

  site_info = read_info_dict(sid=sid)

  for data_type in site_info.keys(): # e.g., GIC, B

    if data_types is not None and data_type not in data_types:
      # Skip this data type if not in requested data_types to plot.
      logger.info(f"  Not reading '{sid}/{data_type}' data type data b/c not in requested data_types = {data_types}.")
      continue

    if data_type not in site_info:
      # This will occur if data_types is given and site does not have a
      # data_type in data_types.
      logger.warning(f"  Requesested data_type = {data_type} not available at site '{sid}'. Skipping.")
      continue

    data_classes = site_info[data_type].keys()

    for data_class in data_classes: # e.g., measured, calculated

      data_sources = site_info[data_type][data_class]

      for data_source in data_sources.keys():

        logger.info(f"  Reading '{data_type}/{data_class}/{data_source}' data for event '{event}'")
        orig = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
        if sid in CONFIG['single_phase_sids'] and data_type == 'GIC':
          logger.info(f"    Multiplying GIC data by 3 to account for single-phase transformer.")
          orig['data'] = orig['data'] * 3
        site_info[data_type][data_class][data_source]['original'] = orig
        # Check returned data object
        if _output_error(orig, logger):
          site_info[data_type][data_class][data_source][sid]['automated_error'] = orig['error']
          continue
        
        #TODO: Make resampling optional setting in config file
        resample_msg = "Resample to 1m aves and NaN pad or trim to start/stop."
        data_mod = orig['data'].copy()
        if 'automated_error' not in site_info[data_type][data_class][data_source][sid].keys():
          add_errors = True
        if add_errors:
          if data_type == 'GIC' and data_class == 'measured':
            from swerve import filter
            logger.info('    Running automated error checks on GIC measured data')
            data_filtered, site_info[data_type][data_class][data_source][sid]['automated_error'], corrections = filter(orig, start, stop, logger=logger)
            data_mod = data_filtered['data']
            resample_msg = corrections + '\n' + resample_msg
          else:
            site_info[data_type][data_class][data_source][sid]['automated_error'] = None
            #TODO: filter for B, DMM data?
        if data_type == 'B' and data_class == 'measured':
            logger.info(f'    Remove baseline then {resample_msg}')
            for i in range(3):
              # TODO: Get IGRF value instead of using first value?
              first_valid_idx = numpy.where(~numpy.isnan(orig["data"][:, i]))[0][0]
              baseline = orig["data"][first_valid_idx, i]
              data_mod[:,i] = orig["data"][:, i] - baseline
        else:
          logger.info(f"    {resample_msg}")

        labels = orig['labels']
        if data_type == 'B':
          # Add column with horizontal magnitude
          h = data_mod[:,0]**2 + data_mod[:,1]**2
          data_mod = numpy.hstack((data_mod, numpy.sqrt(h).reshape(-1, 1)))
          labels.append('B_H')

        modified = {'modification': resample_msg}
        try:
          time_m, data_m = resample(orig["time"], data_mod, start, stop, ave='60s', **resample_kwargs)
          modified['time'] = time_m
          modified['data'] = data_m
          modified['unit'] = orig['unit']
          modified['labels'] = labels
        except Exception as e:
          modified['error'] = str(e)
          logger.error(f"    Error resampling data: {modified['error']}")

        site_info[data_type][data_class][data_source]['modified'] = modified

        file_name = f'{data_type}_{data_class}_{data_source}.pkl'
        file_name = os.path.join(CONFIG['dirs']['processed'], event, 'sites', sidx, 'data', file_name)
        _write_pkl(file_name, site_info[data_type][data_class], logger, indent= ' '*4)

  _write_pkl(site_all_file, site_info, logger, indent=' '*2)

  return site_info


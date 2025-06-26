def site_stats(sid, data, data_types=None, logger=None):

  if logger is None:
    from swerve import LOG_KWARGS, logger
    logger = logger(**LOG_KWARGS)

  for data_type in data.keys(): # e.g., GIC, B
    if data_types is not None and data_type not in data_types:
      continue

    for data_class in data[data_type].keys(): # e.g., measured, calculated
      for data_source in data[data_type][data_class].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        if 'modified' in data[data_type][data_class][data_source]:
          if 'error' not in data[data_type][data_class][data_source]['modified']:
            data_meas = data[data_type][data_class][data_source]['modified']['data']
            stats_summary = _stats_summary(data_meas, logger)
            logger.info(f"  Stats for {data_type}/{data_class}/{data_source}:")
            logger.info(f"    {stats_summary}")

def _stats_calculated(data_meas, data_calc, logger):
  import numpy as np
  valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
  stats_nan = {
          'std': np.nan,
          'cc': np.nan,
          'pe': np.nan,
          'n': len(data_calc),
          'n_valid': np.sum(valid)
  }
  if np.sum(valid) < 3:
      logger.warning("  Not enough valid data. Skipping.")
      return stats_nan

  if np.all(data_calc == 0):
      logger.warning("  All calculated data is zero. Skipping.")
      return stats_nan

  std_calc = np.nanstd(data_calc)
  cc = np.corrcoef(data_meas[valid], data_calc[valid])
  if cc[0,1] < 0:
    data_calc = -data_calc
  numer = np.sum((data_meas[valid] - data_calc[valid])**2)
  denom = np.sum((data_meas[valid] - data_meas[valid].mean())**2)
  pe = 1-numer/denom

  return {
          'std': std_calc,
          'cc': cc[0,1],
          'pe': pe,
          'n': len(data_calc),
          'n_valid': np.sum(valid)
  }

def _stats_summary(data_meas, logger):
  import numpy as np
  return {
          'std': np.nanstd(data_meas),
          '|max|': np.abs(np.nanmax(data_meas)),
          'n': len(data_meas),
          'n_valid': np.sum(~np.isnan(data_meas))
  }

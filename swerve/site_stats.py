def site_stats(sid, data, data_types=None, logger=None):

  import utilrsw

  if logger is None:
    from swerve import LOG_KWARGS, logger
    logger = logger(**LOG_KWARGS)

  logger.info(f"Computing stats for '{sid}' data")

  all_stats = {}
  for data_type in data.keys(): # e.g., GIC, B

    if data_types is not None and data_type not in data_types:
      # Skip this data type if not in requested data_types to plot.
      logger.info(f"  Not computing stats for '{sid}/{data_type}' data type b/c not in requested data_types = {data_types}.")
      continue

    for data_class in data[data_type].keys(): # e.g., measured, calculated

      for data_source in data[data_type][data_class].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        all_stats[f"{data_type}/{data_class}/{data_source}"] = {}
        if 'modified' in data[data_type][data_class][data_source]:
          if 'error' not in data[data_type][data_class][data_source]['modified']:
            data_modified = data[data_type][data_class][data_source]['modified']['data']
            stats = _stats(data_modified, logger)

            key = f"{data_type}/{data_class}/{data_source}"
            all_stats[key]['stats'] = stats
            data[data_type][data_class][data_source]['modified']['stats'] = stats

            logger.info(f"  Stats for {data_type}/{data_class}/{data_source}:")
            logger.info(f"\n{utilrsw.format_dict(stats, indent=4)}")

    if set(("calculated", "measured")).issubset(data[data_type]):
      # Both measured and calculated data are keys in data[data_type].

      # Always use the first data source for measured data.
      data_source_measured = list(data[data_type]['measured'].keys())[0]
      if 'modified' in data[data_type]['measured'][data_source_measured]:
        if 'error' in data[data_type]['measured'][data_source_measured]['modified']:
          msg = f"  Skipping stats for {data_type} due to error in measured modified data."
          logger.warning(msg)
          continue
        data_measured = data[data_type]['measured'][data_source_measured]['modified']['data']

      for data_source_calculated in data[data_type]['calculated'].keys(): # e.g., TVA, GMU, NERC, SWMF, OpenGGCM
        if 'modified' in data[data_type]['calculated'][data_source_calculated]:
          if 'error' in data[data_type]['calculated'][data_source_calculated]['modified']:
            logger.warning(f"  Skipping stats for {data_type} due to error in calculated modified data.")
            continue

        data_calculated = data[data_type]['calculated'][data_source_calculated]['modified']['data']
        metrics = _metrics(data_measured, data_calculated, logger)

        key = f"{data_type}/calculated/{data_source_calculated}"
        all_stats[key]['metrics'] = metrics
        data[data_type]['calculated'][data_source_calculated]['modified']['metrics'] = metrics

        logger.info(f"  Metrics for {key}:")
        logger.info(f"\n{utilrsw.format_dict(metrics, indent=4)}")

  return all_stats

def _stats(data_meas, logger):
  import numpy as np
  return {
          'std': np.nanstd(data_meas, axis=0),
          'ave': np.nanmean(data_meas, axis=0),
          'min': np.min(np.nanmax(data_meas, axis=0)),
          'max': np.min(np.nanmin(data_meas, axis=0)),
          'n': len(data_meas),
          'n_valid': np.sum(~np.isnan(data_meas), axis=0),
  }

def _metrics(data_meas, data_calc, logger):
  import numpy as np

  if len(data_meas.shape) > 1:
    # Compute metrics for each column
    metrics_combined = {}
    for j in range(data_meas.shape[1]):
      metrics_column = _metrics(data_meas[:, j], data_calc[:, j], logger)
      if j == 0:
        metrics_combined = metrics_column
        for key in metrics_column.keys():
          metrics_combined[key] = [metrics_column[key]]
      else:
        for key in metrics_column.keys():
          metrics_combined[key].append(metrics_column[key])

    for key in metrics_combined.keys():
        metrics_combined[key] = np.array(metrics_combined[key])
        if len(metrics_combined[key].shape) > 1:
          metrics_combined[key] = metrics_combined[key].T

    return metrics_combined

  # Compute metrics for a single column
  valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
  stats_nan = {
          'rmse': np.nan,
          'cc': np.nan,
          'pe': np.nan,
          'n_valid': np.sum(valid)
  }
  if np.sum(valid) < 3:
      logger.warning("  Not enough valid data. Skipping.")
      return stats_nan

  if np.all(data_calc == 0):
      logger.warning("  All calculated data is zero. Skipping.")
      return stats_nan

  cc = np.corrcoef(data_meas[valid], data_calc[valid])
  if cc[0,1] < 0:
    data_calc = -data_calc
  numer = np.sum((data_meas[valid] - data_calc[valid])**2)
  denom = np.sum((data_meas[valid] - data_meas[valid].mean())**2)
  pe = 1-numer/denom

  err = data_meas[valid] - data_calc[valid]
  return {
          'err_rms': np.sqrt(np.mean((data_meas[valid] - data_calc[valid])**2)),
          'err_ave': np.mean(err),
          'err': err.flatten(),
          'cc': cc[0,1],
          'pe': pe,
          'valid': valid,
          'n_valid': np.sum(valid)
  }

def cadence(time, logger=None, logger_indent=0):
  import numpy
  if not isinstance(time, numpy.ndarray):
    time = numpy.array(time)
  if not numpy.issubdtype(time.dtype, numpy.datetime64):
    time = time.astype('datetime64[ns]')

  # Compute time differences in seconds between consecutive time points
  dts = numpy.diff(time)

  # Find unique time differences and their counts
  dts_uniq, counts = numpy.unique(dts, return_counts=True)
  if logger is not None:
    logger.info(f"{logger_indent}  Uniq dts [ns] = {dts_uniq}")
    logger.info(f"{logger_indent}  Uniq dts [#]  = {len(dts_uniq)}")

  dts_uniq.sort()
  return dts_uniq

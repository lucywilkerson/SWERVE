def resample(time, data, start, stop, ave=None, logger=None, logger_indent=0):

  def print_info0():
    if logger is None:
      return
    logger.info(f"{logger_indent}Given")
    logger.info(f"{logger_indent}  start         = {start}")
    logger.info(f"{logger_indent}  stop          = {stop}")
    logger.info(f"{logger_indent}  ave           = {ave}")

  def print_info(time, data, msg=None):
    if logger is None:
      return
    if msg is not None:
      logger.info(f"{logger_indent}{msg}")

    logger.info(f"{logger_indent}  data.shape    = {data.shape}")
    logger.info(f"{logger_indent}  nanmean(data) = {numpy.nanmean(data)}")
    logger.info(f"{logger_indent}  time.shape    = {time.shape}")
    logger.info(f"{logger_indent}  time[0]       = {time[0]}")
    logger.info(f"{logger_indent}  data[0]       = {data[0]}")
    if time.shape[0] > 1:
      logger.info(f"{logger_indent}  time[1]       = {time[1]}")
      logger.info(f"{logger_indent}  data[1]       = {data[1]}")
    if time.shape[0] > 2:
      logger.info(f"{logger_indent}  time[-2]      = {time[-2]}")
      logger.info(f"{logger_indent}  data[-2]      = {data[-2]}")
    logger.info(f"{logger_indent}  time[-1]      = {time[-1]}")
    logger.info(f"{logger_indent}  data[-1]      = {data[-1]}")

    _ = cadence(time, data, logger=logger, logger_indent=logger_indent)

  import numpy
  import pandas

  from swerve import cadence
  from swerve import subset

  if logger_indent == 0:
    logger_indent = ''
  else:
    logger_indent = ' ' * logger_indent

  if time.shape[0] != data.shape[0]:
    raise ValueError(f"Time and data lengths do not match: {len(time)} != {len(data)}")

  if logger:
    print_info0()
    print_info(time, data)

  if not isinstance(time, numpy.ndarray):
    time = numpy.array(time)
  if not numpy.issubdtype(time.dtype, numpy.datetime64):
    time = time.astype('datetime64[ns]')
  dts_uniq = cadence(time, data, logger=logger, logger_indent=logger_indent)
  if len(dts_uniq) > 1:
    raise ValueError(f"Time steps are not uniform. Unique dts [s]: {dts_uniq}")
  dts_uniq = dts_uniq[0]  # Unique time difference in nanoseconds.

  time, data = subset(time, data, start, stop)
  print_info(time, data, "After subsetting")

  # inclusive='left' excludes the stop time.
  freq = f"{int(dts_uniq)}ns"  # Frequency in nanoseconds.
  date_range = pandas.date_range(start=start, end=stop, freq=freq, inclusive='left')
  #date_range = pandas.date_range(start=start, end=stop, freq=freq)
  datetime_series = pandas.Series(time)
  dfo = pandas.DataFrame(data, index=datetime_series)

  # number of dimensions in data
  nd = len(data.shape)
  assert nd <= 2, f"data.shape = {data.shape} is not (n, ) or (n, m)"
  if nd == 1:
    fill = numpy.full((len(date_range), ), numpy.nan)
  else:
    fill = numpy.full((len(date_range), data.shape[1]), numpy.nan)
  df = pandas.DataFrame(fill, index=date_range)

  df.update(dfo)

  data = df.to_numpy()
  time = df.index.to_pydatetime()
  print_info(time, data, "After padding with NaNs")

  if ave is not None:
    # Seems like there should be an easier way to shift timestamps.
    if isinstance(ave, (int, float)):
      ave = f"{int(ave)}ns"
    ave = pandas.Timedelta(ave)
    df = df.resample(ave).mean()
    df.index = df.index + ave/2 - dts_uniq/2

  data = df.to_numpy()
  time = df.index.to_pydatetime()

  if ave is not None:
    print_info(time, data, "After resampling")

  return time, data


def resample_test():
  start   = datetime.datetime(2024, 5, 10, 0, 0, 0)
  stop    = datetime.datetime(2024, 5, 10, 0, 1, 0)

  # Create list of 1-second timestamps from start to stop (exclusive of stop)
  time = [start + datetime.timedelta(seconds=i) for i in range(int((stop - start).total_seconds()))]
  time = numpy.array(time)
  data = numpy.ones((len(time), 1))
  data = -1 + numpy.cumsum(data, axis=0)

  logger.info(40 * '-')
  logger.info("Testing trimming start")
  start_x = datetime.datetime(2024, 5, 10, 0, 0, 10)
  time_, data_ = resample(time, data, start_x, stop, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop, f"Expected time[-1] = {stop}, got {time_[-1]}"

  logger.info(40 * '-')
  logger.info("Testing trimming stop")
  stop_x = datetime.datetime(2024, 5, 10, 0, 0, 20)
  time_, data_ = resample(time, data, start_x, stop_x, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop_x, f"Expected time[-1] = {stop_x}, got {time_[-1]}"

  logger.info(40 * '-')
  logger.info("Testing padding start")
  start_x = datetime.datetime(2024, 5, 9, 23, 59, 10)
  time_, data_ = resample(time, data, start_x, stop, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop, f"Expected time[-1] = {stop}, got {time_[-1]}"

  logger.info(40 * '-')
  logger.info("Testing padding stop")
  stop_x = datetime.datetime(2024, 5, 10, 0, 1, 20)
  time_, data_ = resample(time, data, start, stop_x, ave=None, logger=logger)
  assert time_[0] == start, f"Expected time[0] = {start}, got {time_[0]}"
  assert time_[-1] == stop_x, f"Expected time[-1] = {stop}, got {time_[-1]}"

  logger.info(40 * '-')
  logger.info("Testing averaging using ave in nanoseconds")
  start_x = datetime.datetime(2024, 5, 10, 0, 0, 10)
  stop_x = datetime.datetime(2024, 5, 10, 0, 0, 29)
  ave = 10
  time_, data_ = resample(time, data, start_x, stop_x, ave=ave*1e9, logger=logger)
  to = start_x + datetime.timedelta(seconds=ave/2 - 0.5)
  tf = stop_x - datetime.timedelta(seconds=ave/2 - 0.5)
  assert time_[0] == to, f"Expected time[0] = {to}, got {time_[0]}"
  assert data_[0] == 14.5, f"Expected time[0] = {to}, got {time_[0]}"
  assert time_[-1] == tf, f"Expected time[-1] = {tf}, got {time_[-1]}"
  assert data_[-1] == 24.5, f"Expected time[0] = {to}, got {time_[0]}"

  logger.info("Testing averaging using ave as string")
  time_, data_ = resample(time, data, start_x, stop_x, ave='10s', logger=logger)
  to = start_x + datetime.timedelta(seconds=ave/2 - 0.5)
  tf = stop_x - datetime.timedelta(seconds=ave/2 - 0.5)
  assert time_[0] == to, f"Expected time[0] = {to}, got {time_[0]}"
  assert data_[0] == 14.5, f"Expected time[0] = {to}, got {time_[0]}"
  assert time_[-1] == tf, f"Expected time[-1] = {tf}, got {time_[-1]}"
  assert data_[-1] == 24.5, f"Expected time[0] = {to}, got {time_[0]}"

  #print(d)


if __name__ == '__main__':
  resample_test()

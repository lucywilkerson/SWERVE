# Execute using
#   python -m swerve.resample_test

def resample_test():
  import datetime
  import numpy
  from swerve import resample

  from swerve import config
  CONFIG = config()
  logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  start   = datetime.datetime(2024, 5, 10, 0, 0, 0)
  stop    = datetime.datetime(2024, 5, 10, 0, 1, 0)

  # Create list of 1-second timestamps from start to stop (exclusive of stop)
  time = [start + datetime.timedelta(seconds=i) for i in range(int((stop - start).total_seconds()))]
  time = numpy.array(time)
  data = numpy.ones((len(time), 1))
  data = -1 + numpy.cumsum(data, axis=0)
  print(time)
  print(data)

  print(40 * '-')
  print("Testing trimming start")
  start_x = datetime.datetime(2024, 5, 10, 0, 0, 10)
  time_, data_ = resample(time, data, start_x, stop, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop, f"Expected time[-1] = {stop}, got {time_[-1]}"

  print(40 * '-')
  print("Testing trimming stop")
  stop_x = datetime.datetime(2024, 5, 10, 0, 0, 20)
  time_, data_ = resample(time, data, start_x, stop_x, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop_x, f"Expected time[-1] = {stop_x}, got {time_[-1]}"

  print(40 * '-')
  print("Testing padding start")
  start_x = datetime.datetime(2024, 5, 9, 23, 59, 10)
  time_, data_ = resample(time, data, start_x, stop, ave=None, logger=logger)
  assert time_[0] == start_x, f"Expected time[0] = {start_x}, got {time_[0]}"
  assert time_[-1] == stop, f"Expected time[-1] = {stop}, got {time_[-1]}"

  print(40 * '-')
  print("Testing padding stop")
  stop_x = datetime.datetime(2024, 5, 10, 0, 1, 20)
  time_, data_ = resample(time, data, start, stop_x, ave=None, logger=logger)
  assert time_[0] == start, f"Expected time[0] = {start}, got {time_[0]}"
  assert time_[-1] == stop_x, f"Expected time[-1] = {stop}, got {time_[-1]}"

  print(40 * '-')
  print("Testing averaging using ave in nanoseconds")
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

  print("Testing averaging using ave as string")
  time_, data_ = resample(time, data, start_x, stop_x, ave='10s', logger=logger)
  to = start_x + datetime.timedelta(seconds=ave/2 - 0.5)
  tf = stop_x - datetime.timedelta(seconds=ave/2 - 0.5)
  assert time_[0] == to, f"Expected time[0] = {to}, got {time_[0]}"
  assert data_[0] == 14.5, f"Expected time[0] = {to}, got {time_[0]}"
  assert time_[-1] == tf, f"Expected time[-1] = {tf}, got {time_[-1]}"
  assert data_[-1] == 24.5, f"Expected time[0] = {to}, got {time_[0]}"

  #print(data)


if __name__ == '__main__':
  resample_test()

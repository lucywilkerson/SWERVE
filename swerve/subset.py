def subset(time, data, start, stop):
  import numpy
  import datetime

  time_list = False
  if not isinstance(time, numpy.ndarray):
    time_list = True
    time = numpy.array(time)

  time_datetime = False
  if isinstance(time[0], datetime.datetime):
    time_datetime = True
    # TODO: Check that all elements are datetime.datetime?

  if not numpy.issubdtype(time.dtype, numpy.datetime64):
    time = time.astype('datetime64[ns]')

  if not isinstance(start, numpy.datetime64):
    start = numpy.datetime64(start, 'ns')
  if not isinstance(stop, numpy.datetime64):
    stop = numpy.datetime64(stop, 'ns')

  idx = numpy.logical_and(time >= start, time <= stop)
  time = time[idx]

  #import pdb; pdb.set_trace()

  if time_datetime:
    # https://stackoverflow.com/questions/72484789/numpy-array-tolist-converts-numpy-datetime64-to-int/72484984#72484984
    # Must cast to milliseconds first so tolist() gives native Python datetime objects
    time = time.astype('datetime64[us]').tolist()

  if not time_list:
    time = numpy.array(time)

  if data.ndim == 1:
    return time, data[idx]

  return time, data[idx, :]

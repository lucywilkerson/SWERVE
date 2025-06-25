def subset(time, data, start, stop):
  import numpy

  time_list = False
  if not isinstance(time, numpy.ndarray):
    time_list = True
    time = numpy.array(time)

  time_datetime = True
  if not numpy.issubdtype(time.dtype, numpy.datetime64):
    time_datetime = False
    time = time.astype('datetime64[ns]')
  if not isinstance(start, numpy.datetime64):
    start = numpy.datetime64(start, 'ns')
  if not isinstance(stop, numpy.datetime64):
    stop = numpy.datetime64(stop, 'ns')

  idx = numpy.logical_and(time >= start, time <= stop)
  time = time[idx]

  if time_datetime:
    time = time.astype('datetime64[us]').tolist()
  if not time_list:
    time = numpy.array(time)

  if data.ndim == 1:
    return time, data[idx]

  return time, data[idx, :]

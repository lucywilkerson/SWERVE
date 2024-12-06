import os
import csv
import json
import numpy
import pandas
import pickle
import datetime

data_dir = os.path.join('..', '2024-AGU-data')

def read(info, sid, data_type, data_class, data_subclass, data_dir):

  data_info = info[sid]['data'][data_type][data_class]

  if data_subclass is None:
    data_dir = os.path.join(data_dir, data_info['dir'])
  else:
    for idx, subclass in enumerate(data_info):
      # Loop through array of objects.
      if subclass['model'] == data_subclass:
        data_info = data_info[idx]
        break
    data_dir = os.path.join(data_dir, data_info['dir'])

  if data_type == 'gic' and data_class == 'measured':
    files = data_info['files']
    if files[0].endswith('.csv'):
      data = []
      time = []
      for file in files:
        file = os.path.join(data_dir, file)
        print(f"    Reading {file}")
        if not os.path.exists(file):
          raise FileNotFoundError(f"File not found: {file}")
        with open(file,'r') as csvfile:
          rows = csv.reader(csvfile, delimiter = ',')
          for row in rows:
              time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
              data.append(float(row[1]))

      return numpy.array(time), numpy.array(data)

  if data_type == 'gic' and data_class == 'calculated':
    files = data_info['files']
    starts = data_info['start']
    time = []
    data = []
    for idx, file in enumerate(files):
      start = starts[idx]
      dto = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
      file = os.path.join(data_dir, file)
      print(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
      d, times = numpy.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
      data.append(d)
      for t in times:
        time.append(dto + datetime.timedelta(seconds=t))

    return numpy.array(time).flatten(), numpy.array(data).flatten()

  if data_type == 'mag' and data_class == 'measured':

      b  = []
      time = []

      file = data_info['files'][0]
      file = os.path.join(data_dir, file)
      print(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

      with open(file,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter = ',')
        next(rows)  # Skip header row.
        for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          b.append([float(row[1]), float(row[2]), float(row[3])])

      return numpy.array(time), numpy.array(b)

  if data_type == 'mag' and data_class == 'calculated':

    if data_subclass == 'swmf':
      df = {}
      for region in ["gap", "iono", "msph"]:
        file = os.path.join(data_dir, data_info['files'][region])
        print(f"    Reading {file}")
        if not os.path.exists(file):
          raise FileNotFoundError(f"File not found: {file}")
        df[region]  = pandas.read_pickle(file)

      bx = df["gap"]['Bn'] + df["iono"]['Bnh'] + df["iono"]['Bnp'] + df["msph"]['Bn']
      by = df["gap"]['Be'] + df["iono"]['Beh'] + df["iono"]['Bep'] + df["msph"]['Be']
      bz = df["gap"]['Bd'] + df["iono"]['Bdh'] + df["iono"]['Bdp'] + df["msph"]['Bd']

      b = numpy.vstack([bx.to_numpy(), by.to_numpy(), bz.to_numpy()])
      time = bx.keys() # Will be the same for all
      time = time.to_pydatetime()

      return time, b.T

    if data_subclass == 'mage':
      # TODO: A single file has data from all sites. Here we read full
      # file and extract data for the requested site. Modify this code
      # so sites dict is cached and used if found.
      file = os.path.join(data_dir, data_info['files'][0])
      print(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

      sites = {}
      with open(file,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter = ',')
        next(rows)  # Skip header row.
        for row in rows:
          site = row[0]
          if site not in sites:
            sites[site] = {"time": [], "data": []}
          sites[site]["time"].append(datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'))
          # TODO: Determine the coordinate system of the data.
          #       Header is
          #          site,time,dBn,dBt,dBp,dBr,glon,glat,mlon,mlat
          sites[site]["data"].append([float(row[2]), float(row[3]), float(row[4])])

      site_requested = info[sid]['name']
      if site_requested not in sites:
        raise ValueError(f"Requested site name = '{site_requested}' associated with site id = '{sid}' not found in {file}")

      return numpy.array(sites[site_requested]["time"]), numpy.array(sites[site_requested]["data"])

def resample(time, data, start, stop, freq, ave=None):

  # inclusive='left' will exclude the stop time.
  date_range = pandas.date_range(start=start, end=stop, freq=freq, inclusive='left')

  datetime_series = pandas.Series(time)
  dfo = pandas.DataFrame(data, index=datetime_series)

  # number of dimensions
  nd = len(data.shape)
  assert nd <= 2, f"data.shape = {data.shape} is not (n, ) or (n, m)"
  if nd == 1:
    fill = numpy.full((len(date_range), ), numpy.nan)
  else:
    fill = numpy.full((len(date_range), data.shape[1]), numpy.nan)
  df = pandas.DataFrame(fill, index=date_range)

  df.update(dfo)

  if ave is not None:
    # Seems like there should be an easier way to shift timestamps.
    # https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series
    df = df.resample(f"{ave}s").mean()
    df.index = df.index + datetime.timedelta(seconds=ave/2)

  data = df.to_numpy()
  # If data.shape is (n, ), return a 1D array.
  # If data.shape is (n, m), return a 2D array.
  if nd == 1: # (n, ) case
    data = data.flatten()
  return df.index.to_pydatetime(), data


fname = os.path.join(data_dir, 'info.json')
with open(fname, 'r') as f:
  print(f"Reading {fname}\n")
  info = json.load(f)

start = datetime.datetime(2024, 5, 10, 0, 0)
stop = datetime.datetime(2024, 5, 13, 0, 0)

data = {}
for sid in info.keys(): # site ids
  data[sid] = {}
  print(f"Reading '{sid}' data")
  data_types = info[sid]['data'].keys()

  for data_type in data_types: # e.g., gic, mag

    data[sid][data_type] = {}
    data_classes = info[sid]['data'][data_type].keys()

    for data_class in data_classes: # e.g., measured, calculated

      if isinstance(info[sid]['data'][data_type][data_class], list):
        data_subclasses = info[sid]['data'][data_type][data_class]
        data[sid][data_type][data_class] = []
        for data_subclass in data_subclasses:
          model = data_subclass["model"]
          print(f"  Reading '{data_type}/{data_class}/{model}' data")
          time, data_ = read(info, sid, data_type, data_class, model, data_dir)
          print(f"    data.shape = {data_.shape}")
          original = {'original': {'time': time, 'data': data_}}
          data[sid][data_type][data_class].append(original)
      else:
        print(f"  Reading '{data_type}/{data_class}' data")
        time, data_ = read(info, sid, data_type, data_class, None, data_dir)
        print(f"    data.shape = {data_.shape}")
        original = {'original': {'time': time, 'data': data_}}
        data[sid][data_type][data_class] = original

      if data_type == 'gic' and data_class == 'measured':
        # Resample to 1-min average.
        print("    Averaging timeseries to 1-min bins")
        time, data_ = resample(time, data_, start, stop, 's', ave=60)
        print(f"    data.shape = {data_.shape}")
        modified = {'time': time, 'data': data_, 'modification': '1-min average'}
        data[sid][data_type][data_class]['modified'] = modified

      if data_type == 'mag' and data_class == 'measured':
        # Remove mean
        print("    Creating timeseries with mean removed")
        data_m = numpy.full(data_.shape, numpy.nan)
        for i in range(3):
          #numpy.nanmean(data_[:,i])
          # TODO: Get IGRF value 
          data_m[:,i] = data_[:,i] - data_[0,i]
        print(f"    data.shape = {data_.shape}")
        modified = {'time': time, 'data': data_m, 'modification': 'mean removed'}
        data[sid][data_type][data_class]['modified'] = modified

fname = os.path.join(data_dir, 'data.pkl')
print(f"\nWriting {fname}")
with open(fname, 'wb') as f:
  pickle.dump(data, f)

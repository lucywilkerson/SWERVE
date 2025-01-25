import os
import csv
import json
import numpy
import pandas
import pickle
import datetime

data_dir = os.path.join('..', '2024-AGU-data')
base_dir = os.path.join(data_dir, '_processed')
all_file = os.path.join(data_dir, '_all', 'all.pkl')
if not os.path.exists(os.path.dirname(all_file)):
  os.makedirs(os.path.dirname(all_file))

def read_nerc(data_dir, fname):
  data = []
  time = []

  file = os.path.join(data_dir, fname)
  print(f"    Reading {file}")
  if not os.path.exists(file):
    raise FileNotFoundError(f"File not found: {file}")
  with open(file, 'r') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    next(rows)  # Skip header row.
    device_id_last = None
    for row in rows:
      device_id = row[0]
      if device_id != device_id_last:
        if device_id_last is not None:
          raise ValueError(f"Multiple device ids found in {file}")
        device_id_last = device_id
      time_p = datetime.datetime.strptime(row[1], '%m/%d/%Y %I:%M:%S %p')
      time.append(time_p)
      cols = []
      for i in range(2, len(row)):
        cols.append(float(row[i]))
      data.append(cols)

  # Print duplicate times
  time = numpy.array(time)
  # e.g., 2024E04_10233.csv, which looks like it is 10-second cadence data
  # but time stamps have the same second value.
  if len(time) != len(numpy.unique(time)):
    print("    Error: duplicate time stamps found")
    return None
    #print(time[numpy.where(numpy.diff(time) == datetime.timedelta(0))])

  data = numpy.array(data)
  if data.shape[1] == 1:
    data = data.flatten()

  return {"time": numpy.array(time).flatten(), "data": data}

def read(info, sid, data_type, data_class, data_source, data_dir):

  data = []
  time = []
  if data_type == 'GIC' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'gic', 'GIC-measured')
    sid = sid.lower().replace(' ','')
    if sid == 'widowscreek':
      sid = f'{sid}2'
    fname = f'gic-{sid}_20240510.csv'
    file = os.path.join(data_dir, fname)
    print(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          data.append(float(row[1]))

    return {"time": numpy.array(time).flatten(), "data": numpy.array(data).flatten()}

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'NERC':
    fname = f'2024E04_{sid}.csv'
    data_dir = os.path.join(data_dir, 'nerc', 'gic')
    return read_nerc(data_dir, fname)

  if data_type == 'GIC' and data_class == 'calculated' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'gic', 'GIC-calculated')
    sid = sid.replace(' ','')
    if sid == 'BullRun':
      sid = 'BullRunXfrm' # BullRun file Xfrm appended to name in file name.
    if sid == 'WidowsCreek':
      sid = f'{sid}2'

    dates = ['20240510', '20240511', '20240512']
    time = []
    data = []
    for date in dates:
      file = f'{date}_{sid}GIC.dat'
      file = os.path.join(data_dir, file)
      print(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
      d, times = numpy.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
      data.append(d)
      dto = datetime.datetime.strptime(date, '%Y%m%d')
      for t in times:
        time.append(dto + datetime.timedelta(seconds=t))

    return {"time": numpy.array(time).flatten(), "data": numpy.array(data).flatten()}

  if data_type == 'B' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'mag')
    sid = sid.lower().replace(' ','')

    data  = []
    time = []

    file = os.path.join(data_dir, f'{sid}_mag_20240509.csv')
    print(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")

    with open(file,'r') as csvfile:
      rows = csv.reader(csvfile, delimiter = ',')
      next(rows)  # Skip header row.
      for row in rows:
        time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        data.append([float(row[1]), float(row[2]), float(row[3])])

    data = numpy.array(data)
    time = numpy.array(time)
    return {"time": time, "data": data}

  if data_type == 'B' and data_class == 'measured' and data_source == 'NERC':
    fname = f'2024E04_{sid}.csv'
    data_dir = os.path.join(data_dir, 'nerc', 'mag')
    return read_nerc(data_dir, fname)

  if data_type == 'B' and data_class == 'calculated' and data_source == 'SWMF':

    sid = sid.replace(' ','')
    data_dir = os.path.join(data_dir, 'swmf', sid.lower())

    df = {}
    for region in ["gap", "iono", "msph"]:
      file = os.path.join(data_dir, f'dB_bs_{region}-{sid}.pkl')
      print(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
      df[region]  = pandas.read_pickle(file)

    bx = df["gap"]['Bn'] + df["iono"]['Bnh'] + df["iono"]['Bnp'] + df["msph"]['Bn']
    by = df["gap"]['Be'] + df["iono"]['Beh'] + df["iono"]['Bep'] + df["msph"]['Be']
    bz = df["gap"]['Bd'] + df["iono"]['Bdh'] + df["iono"]['Bdp'] + df["msph"]['Bd']


    data = numpy.vstack([bx.to_numpy(), by.to_numpy(), bz.to_numpy()])
    time = bx.keys() # Will be the same for all
    time = time.to_pydatetime()

    return {"time": time, "data": data.T}

  if data_type == 'B' and data_class == 'calculated' and data_source == 'MAGE':
    # TODO: A single file has data from all sites. Here we read full
    # file and extract data for the requested site. Modify this code
    # so sites dict is cached and used if found.
    file = os.path.join(data_dir, 'mage', 'TVAinterpdf.csv')
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
        # Header is
        # site,time,dBn,dBt,dBp,dBr,glon,glat,mlon,mlat
        # From Mike Wiltberger:
        # As a reminder I will point you to our documentation for the kaipy package which
        # includes information on the structure of SuperMAGE interpolated dataframe.
        # Here's the summary from that documentation.
        # - dBn: Interpolated northward deflection (dot product of dB and minus the theta unit vector)
        # - dBt: Interpolated magnetic theta component.
        # - dBp: Interpolated magnetic phi component.
        # - dBr: Interpolated magnetic radial component
        # It doesn't help that we didn't use a consistent naming convention with the SuperMAG data.
        # Here's the relevant mapping BNm - dBt, BEm - dBp, and BZm - dBr.
        # I've attached a reference plot as an example to this message.
        sites[site]["data"].append([float(row[2]), float(row[3]), float(row[4]), float(row[5])])

    if sid not in sites:
      raise ValueError(f"Requested site name = '{sid}' associated with site id = '{sid}' not found in {file}")

    time = numpy.array(sites[sid]["time"])
    data = numpy.array(sites[sid]["data"])
    return {"time": time, "data": data}

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
  return {"time": df.index.to_pydatetime(), "data": data}


fname = os.path.join('info', 'info_dict.json')
with open(fname, 'r') as f:
  print(f"Reading {fname}\n")
  info = json.load(f)

start = datetime.datetime(2024, 5, 10, 0, 0)
stop = datetime.datetime(2024, 5, 13, 0, 0)

data = {}
sids = info.keys()
#sids = ['Union', 'Montgomery', 'Widows Creek', 'Bull Run']
sids = ['10052', '10064']
#sids = ['10052', '10064']
#sids = ['Bull Run', '10052', '50100']
#sids = ['10233']

for sid in sids: # site ids
  data[sid] = {}

  print(f"Reading '{sid}' data")
  data_types = info[sid].keys()

  for data_type in data_types: # e.g., GIC, B

    data[sid][data_type] = {}
    data_classes = info[sid][data_type].keys()

    for data_class in data_classes: # e.g., measured, calculated

      data_sources = info[sid][data_type][data_class]
      data[sid][data_type][data_class] = []
      for data_source in data_sources:
        print(f"  Reading '{data_type}/{data_class}/{data_source}' data")
        d = read(info, sid, data_type, data_class, data_source, data_dir)
        if d is None:
          print("    Skipping")
          data[sid][data_type][data_class].append(None)
          continue
        print(f"    data.shape = {d["data"].shape}")
        original = {'original': d}
        data[sid][data_type][data_class].append(original)

        if data_type == 'GIC' and data_class == 'measured':
          # Resample to 1-min average.
          print("    Averaging timeseries to 1-min bins")
          d = resample(d["time"], d["data"], start, stop, 's', ave=60)
          print(f"    data.shape = {d["data"].shape}")
          modified = {**d, 'modification': '1-min average'}
          # Assumes only one data source.
          data[sid][data_type][data_class][0]['modified'] = modified

        if data_type == 'B' and data_class == 'measured':
          # Remove mean
          print("    Creating timeseries with mean removed")
          data_m = numpy.full(d["data"].shape, numpy.nan)
          for i in range(3):
            # TODO: Get IGRF value
            data_m[:,i] = d["data"][:,i] - d["data"][0,i]
          print(f"    data.shape = {d["data"].shape}")
          modified = {'time': d["time"], 'data': data_m, 'modification': 'mean removed'}
          # Assumes only one data source.
          data[sid][data_type][data_class][0]['modified'] = modified

        sidx = sid.lower().replace(' ', '')
        fname = f'{data_type}_{data_class}_{data_source}.pkl'
        fname = os.path.join(base_dir, sidx, fname)
        if not os.path.exists(os.path.dirname(fname)):
          os.makedirs(os.path.dirname(fname))

        with open(fname, 'wb') as f:
          print(f"    Writing {fname}")
          pickle.dump(data[sid][data_type][data_class], f)

print(f"\nWriting {all_file}")
with open(all_file, 'wb') as f:
  pickle.dump(data, f)

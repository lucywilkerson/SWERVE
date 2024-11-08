import os
import csv
import json
import numpy
import pickle
import datetime

# .dat files are 1-min cadence and are model-predicted GICs (what model?)
# .csv is 1-min cadence and is observed GIC (from where?)
data_dir = os.path.join('..', '..', '2024-AGU-data')
data_dir_gic = os.path.join(data_dir, 'gic')

with open(os.path.join(data_dir, 'info.json'), 'r') as f:
  gic_all = json.load(f)

def read(info, starts=None, data_dir=data_dir):

  files = info['files']
  if files[0].endswith('.csv'):
    data = []
    time = []
    for file in files:
      file = os.path.join(data_dir, file)
      print(f"Reading {file}")
      with open(file,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter = ',')
        for row in rows:
            time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
            data.append(float(row[1]))

    return numpy.array(data), time

  time = []
  data = []
  for idx, file in enumerate(files):
    start = info['start'][idx]
    dto = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    print(f"Reading {file}")
    file = os.path.join(data_dir, file)
    d, times = numpy.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
    data.append(d)
    for t in times:
      time.append(dto + datetime.timedelta(seconds=t))

  data = numpy.array(data).flatten()
  time = numpy.array(time).flatten()
  return data, time

for key, value in gic_all.items():
  info = gic_all[key]
  for file in value['files']:
    path = os.path.join(data_dir_gic, file)
    info['data'], info['time'] = read(info, data_dir=data_dir_gic)

fname = os.path.join(data_dir_gic, 'gic_all.pkl')
print(f"Writing {fname}")
with open(fname, 'wb') as f:
  pickle.dump(gic_all, f)

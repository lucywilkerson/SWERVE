import os
import json
import numpy
import pickle
import pandas
import datetime

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times'
plt.rcParams['figure.dpi'] = 150

data_dir = os.path.join('..', '2024-AGU-data')

fname = os.path.join(data_dir, 'info.json')
with open(fname, 'r') as f:
  print(f"Reading {fname}\n")
  info = json.load(f)

fname = os.path.join(data_dir, 'data.pkl')
print(f"\nReading {fname}")
with open(fname, 'rb') as f:
  data = pickle.load(f)

start = datetime.datetime(2024, 5, 10, 12, 0)
stop = datetime.datetime(2024, 5, 12, 12, 0)
#start = datetime.datetime(2024, 5, 10, 17, 0)
#stop = datetime.datetime(2024, 5, 10, 17, 2)

def subset(time, data, start, stop):
  idx = numpy.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

def plot(time, data, title, ylabel, legend, time_r, data_r):
  plt.title(title)
  plt.plot(time, data, linewidth=1)

  if ylabel is not None:
    plt.ylabel(ylabel)

  if time_r is not None:
    plt.plot(time_r, data_r, linewidth=0.4)

  if legend is not None:
    plt.legend(legend)

  plt.xlim(start, stop)
  plt.grid()


# Compare select quantities on the same figure
for sid in info.keys(): # site ids
  time_meas = data[sid]['gic']['measured']['modified']['time']
  data_meas = data[sid]['gic']['measured']['modified']['data']
  time_calc = data[sid]['gic']['calculated']['original']['time']
  data_calc = data[sid]['gic']['calculated']['original']['data']

  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  time_calc, data_calc = subset(time_calc, data_calc, start, stop)
  if sid == 'bullrun':
    # TODO: Put in info.json or find sign with best calc/meas agreement
    data_calc = -data_calc

  plt.grid()
  plt.plot(time_meas, data_meas, 'k', label='Measured', linewidth=1)
  plt.plot(time_calc, data_calc, 'b', label='Calculated', linewidth=0.4)
  plt.plot(time_calc, data_meas-data_calc-20, 'k', label='error - 20 [nT]', linewidth=0.5)
  plt.legend()
  plt.ylabel('GIC [A]')

  plt.figure()
  bbox = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
  cc = numpy.corrcoef(data_meas, data_calc)
  plt.plot(data_meas, data_calc, 'k.', markersize=1)
  plt.plot([-20, 20], [-20, 20], 'k', linewidth=0.5)
  plt.text(0, 21, f"cc = {cc[0,1]:.2f}", bbox=bbox, horizontalalignment='center')
  plt.xlabel('Measured GIC [A]')
  plt.ylabel('Calculated GIC [A]')
  plt.grid()

if False:
  # Plot all data on separate figures
  for sid in info.keys(): # site ids
    for data_type in info[sid]['data'].keys(): # e.g., gic, mag
      for data_class in info[sid]['data'][data_type].keys(): # e.g., measured, calculated
        #print(f"  {sid}/{data_type}/{data_class}")
        title = f"{sid}/{data_type}/{data_class}"

        # Extract time and data arrays. "o" for original.
        time_o = data[sid][data_type][data_class]['original']['time']
        data_o = data[sid][data_type][data_class]['original']['data']

        # Subset to desired time range
        time_o, data_o = subset(time_o, data_o, start, stop)

        ylabel = None
        if data_type == 'gic':
          ylabel = 'GIC [A]'

        legend = None
        if data_type == 'mag':
          legend = info[sid]['data'][data_type][data_class]['legend']
          ylabel = '[nT]'

        time_m, data_m = None, None

        if data_type == 'gic' and data_class == 'measured':
          time_m = data[sid][data_type][data_class]['modified']['time']
          data_m = data[sid][data_type][data_class]['modified']['data']
          time_m, data_m = subset(time_o, data_o, start, stop)
          legend = ['1-sec orig', '1-min avg']

        plot(time_o, data_o, title, ylabel, legend, time_m, data_m)
        plt.show()

        if data_type == 'mag' and data_class == 'measured':
          #plt.figure()
          time_m = data[sid][data_type][data_class]['modified']['time']
          data_m = data[sid][data_type][data_class]['modified']['data']
          #import pdb; pdb.set_trace()
          time_m, data_m = subset(time_m, data_m, start, stop)
          legend = info[sid]['data'][data_type][data_class]['legend']
          ylabel = '[nT]'
          title = f"{title} with mean removed"
          #print(data_m.shape)
          plot(time_m, data_m, title, ylabel, legend, None, None)
          plt.show()


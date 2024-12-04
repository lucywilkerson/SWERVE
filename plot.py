import os
import sys
import json
import numpy
import pickle
import datetime

from datetick import datetick

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

data_dir = os.path.join('..', '2024-AGU-data')
sids = None # If none, plot all sites
#sids = ['widowscreek']
#sids = ['bullrun', 'widowscreek']
#sids = ['bullrun']

#%config InlineBackend.figure_formats = ['png']
#try:
  #%config InlineBackend.figure_formats = ['svg']
#except:
#  pass

start = datetime.datetime(2024, 5, 10, 12, 0)
stop = datetime.datetime(2024, 5, 13, 0, 0)
#start = datetime.datetime(2024, 5, 10, 17, 0)
#stop = datetime.datetime(2024, 5, 10, 17, 2)

def read(data_dir, sid=None):
  fname = os.path.join(data_dir, 'info.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    info = json.load(f)

  fname = os.path.join(data_dir, 'data.pkl')
  print(f"Reading {fname}")
  with open(fname, 'rb') as f:
    data = pickle.load(f)

  return info, data

def subset(time, data, start, stop):
  idx = numpy.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

def savefig(sid, fname, data_dir=".", fmts=['png']):
  fdir = os.path.join(data_dir, 'processed', sid)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def compare_gic(info, data, sid):

  time_meas = data[sid]['gic']['measured']['modified']['time']
  data_meas = data[sid]['gic']['measured']['modified']['data']
  time_calc = data[sid]['gic']['calculated']['original']['time']
  data_calc = data[sid]['gic']['calculated']['original']['data']

  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  time_calc, data_calc = subset(time_calc, data_calc, start, stop)
  #if sid == 'bullrun' or sid == 'union':
    # TODO: Put in info.json or find sign with best calc/meas agreement
  data_calc = -data_calc

  name = info[sid]['name']

  plt.figure()
  error_shift = 50
  yticks = numpy.arange(-80, 30, 10)
  labels = []
  for ytick in yticks:
    if ytick < -30:
      labels.append(str(ytick+error_shift))
    else:
      labels.append(str(ytick))

  plt.axhline(y=-35, color='w', linestyle='-', linewidth=10, xmin=0, xmax=1)
  plt.axhline(y=-80, color='w', linestyle='-', linewidth=10, xmin=0, xmax=1)
  plt.title(name)
  plt.grid()
  plt.plot()
  plt.plot(time_meas, data_meas, 'k', label='GIC Measured', linewidth=1)
  plt.plot(time_calc, data_calc, 'b', label='GIC Calculated', linewidth=0.4)
  # TODO: Put error on subplot
  plt.plot(time_calc, data_meas-data_calc-error_shift, color=3*[0.3], label=f'Error', linewidth=0.5)
  plt.legend()
  plt.ylabel('[A]', rotation=0, labelpad=10)
  plt.yticks(yticks, labels=labels)
  plt.ylim(-80, 30)
  datetick()
  # get the legend object
  leg = plt.gca().legend()

  # change the line width for the legend
  for line in leg.get_lines():
      line.set_linewidth(1.5)

  savefig(sid, 'gic-timeseries-meas-calc-error', data_dir=data_dir)

  plt.figure()
  cc = numpy.corrcoef(data_meas, data_calc)
  pe = 1-numpy.var(data_meas-data_calc)/numpy.var(data_meas)
  text = f"cc = {cc[0,1]:.2f} | pe = {pe:.2f}"
  text_kwargs = {
    'horizontalalignment': 'center',
    'verticalalignment': 'center',
    'bbox': {
      "boxstyle": "round,pad=0.3",
      "edgecolor": "black",
      "facecolor": "white",
      "linewidth": 0.5
    }
  }
  plt.title(name)
  plt.plot([-40, 40], [-40, 40], color=3*[0.6], linewidth=0.5)
  plt.plot(data_meas, data_calc, 'k.', markersize=1)
  # TODO: Compute limits based on data
  plt.text(-30, 40, text, **text_kwargs)
  plt.xlabel('Measured GIC [A]')
  plt.ylabel('Calculated GIC [A]')
  plt.grid()
  savefig(sid, 'gic-correlation-meas_vs_calc', data_dir=data_dir)

  plt.figure()
  #plt.title(name)
  bl = -10
  bu = 10
  bins_c = numpy.arange(bl, bu+1, 1)
  bins_e = numpy.arange(bl-0.5, bu+1, 1)
  n_e, _ = numpy.histogram(data_meas-data_calc, bins=bins_e)
  plt.bar(bins_c, n_e/sum(n_e), width=0.98, color='k')
  plt.xticks(bins_c[0::2])
  plt.xticks(fontsize=18)
  plt.xlabel('(Measured - Calculated) GIC [A]', fontsize=18)
  plt.xlim(bl-0.5, bu+0.5)
  plt.yticks(fontsize=18)
  plt.ylabel('Probability', fontsize=18)
  plt.grid(axis='y', color=[0.2,0.2,0.2], linewidth=0.2)
  savefig(sid, 'gic-histogram-meas-calc', data_dir=data_dir)

def compare_dB(info, data, sid):
  name = info[sid]['name']

  time_meas = data[sid]['mag']['measured']['modified']['time']
  data_meas = data[sid]['mag']['measured']['modified']['data']
  time_calc = data[sid]['mag']['calculated-swmf']['original']['time']
  data_calc = data[sid]['mag']['calculated-swmf']['original']['data']

  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  time_calc, data_calc = subset(time_calc, data_calc, start, stop)

  data_meas = numpy.linalg.norm(data_meas, axis=1)
  data_calc = numpy.linalg.norm(data_calc, axis=1)
  plt.figure()
  plt.title(name)
  plt.plot(time_meas, data_meas, 'k', linewidth=1, label='Measured')
  plt.plot(time_calc, data_calc, 'b', linewidth=1, label='SWMF')
  plt.ylabel(r'$\Delta$B [nT]')
  datetick()
  plt.legend()
  plt.grid()

def plot_original(info, data, sid):

  # Plot original data on separate figures

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
    datetick()
    plt.grid()

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
        time_m, data_m = subset(time_m, data_m, start, stop)
        legend = ['1-sec orig', '1-min avg']

      plt.figure()
      plot(time_o, data_o, title, ylabel, legend, time_m, data_m)
      savefig(sid, f'{data_type}-{data_class}', data_dir=data_dir)

      if data_type == 'mag' and data_class == 'measured':

        plt.figure()
        time_m = data[sid][data_type][data_class]['modified']['time']
        data_m = data[sid][data_type][data_class]['modified']['data']
        #import pdb; pdb.set_trace()
        time_m, data_m = subset(time_m, data_m, start, stop)
        legend = info[sid]['data'][data_type][data_class]['legend']
        ylabel = '[nT]'
        title = f"{title} with mean removed"

        plot(time_m, data_m, title, ylabel, legend, None, None)
        savefig(sid, f'{data_type}-{data_class}-zero_mean', data_dir=data_dir)

info, data = read(data_dir)

if sids is None:
  sids = info.keys()

for sid in sids: # site ids
  if sid not in info.keys():
    raise ValueError(f"Site '{sid}' not found in info.json")

  mag_types = info[sid]['data']['mag'].keys()
  if 'measured' in mag_types and 'calculated-swmf' in mag_types:
    compare_dB(info, data, sid)
  continue

  plot_original(info, data, sid)

  gic_types = info[sid]['data']['gic'].keys()
  if 'measured' and 'calculated' in gic_types:
    compare_gic(info, data, sid)

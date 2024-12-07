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
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-AGU-data')
sids = None # If none, plot all sites
sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']

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
  fname = os.path.join('info', 'info_data.json')
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
  fdir = os.path.join(data_dir, 'processed', sid.lower().replace(' ', ''))
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def compare_gic(info, data, sid):

  time_meas = data[sid]['GIC']['measured']['modified']['time']
  data_meas = data[sid]['GIC']['measured']['modified']['data']
  time_calc = data[sid]['GIC']['calculated']['original']['time']
  data_calc = data[sid]['GIC']['calculated']['original']['data']

  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  time_calc, data_calc = subset(time_calc, data_calc, start, stop)

  # TODO: Document why this is necessary
  data_calc = -data_calc

  plt.figure()
  error_shift = 50
  yticks = numpy.arange(-80, 30, 10)
  labels = []
  for ytick in yticks:
    if ytick < -30:
      labels.append(str(ytick+error_shift))
    else:
      labels.append(str(ytick))

  kwargs = {"color": 'w', "linestyle": '-', "linewidth": 10, "xmin": 0, "xmax": 1}
  plt.axhline(y=-35, **kwargs)
  plt.axhline(y=-80, **kwargs)
  plt.title(sid)
  plt.grid()
  plt.plot()
  plt.plot(time_meas, data_meas, 'k', label='GIC Measured', linewidth=1)
  plt.plot(time_calc, data_calc, 'b', label='GIC Calculated', linewidth=0.4)

  plt.plot(time_calc, data_meas-data_calc-error_shift, color=3*[0.3], label='Error', linewidth=0.5)
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

  savefig(sid, 'GIC-compare-timeseries', data_dir=data_dir)

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
  plt.title(sid)
  plt.plot([-40, 40], [-40, 40], color=3*[0.6], linewidth=0.5)
  plt.plot(data_meas, data_calc, 'k.', markersize=1)
  # TODO: Compute limits based on data
  plt.text(-30, 40, text, **text_kwargs)
  plt.xlabel('Measured GIC [A]')
  plt.ylabel('Calculated GIC [A]')
  plt.grid()
  savefig(sid, 'GIC-compare-correlation', data_dir=data_dir)

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
  savefig(sid, 'GIC-histogram-meas-calc', data_dir=data_dir)

def compare_dB(info, data, sid):
  time_meas = data[sid]['B']['measured']['modified']['time']
  data_meas = data[sid]['B']['measured']['modified']['data']
  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  data_meas = numpy.linalg.norm(data_meas, axis=1)

  model_names = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g']
  model_names = []
  for idx, data_subclass in enumerate(info[sid]['B']['calculated']):
    model_names.append(data_subclass)
    time_calc = data[sid]['B']['calculated'][idx]['original']['time']
    data_calc = data[sid]['B']['calculated'][idx]['original']['data']
    time_calc, data_calc = subset(time_calc, data_calc, start, stop)
    data_calc = numpy.linalg.norm(data_calc[:,0:2], axis=1)
    time_calcs.append(time_calc)
    data_calcs.append(data_calc)

  plt.figure()
  plt.title(sid)
  plt.plot(time_meas, data_meas, 'k', linewidth=1, label='Measured')
  for idx in range(len(model_names)):
    label = model_names[idx].upper()
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)
  plt.ylabel(r'$\Delta B_H$ [nT]')
  datetick()
  plt.legend()
  plt.grid()

  ax = plt.gca()
  aspect_ratio = 0.5
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*aspect_ratio)

  savefig(sid, 'B-compare-timeseries', data_dir=data_dir)

def plot_original(info, data, sid, data_type, data_class, model):

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

  #print(f"  {sid}/{data_type}/{data_class}")
  title = f"{sid}/{data_type}/{data_class}"
  mag_legend = ["$\\Delta B_x$", "$\\Delta B_y$", "$\\Delta B_z$"]
  base_name = f'{data_type}-{data_class}'
  if model is not None:
    title = f"{title}/{model}"
    base_name = f'{base_name}-{model}'

  # Extract time and data arrays. "o" for original.
  time_o = data['original']['time']
  data_o = data['original']['data']

  # Subset to desired time range
  time_o, data_o = subset(time_o, data_o, start, stop)

  ylabel = None
  if data_type == 'gic':
    ylabel = 'GIC [A]'

  legend = None
  if data_type == 'mag':
    legend = mag_legend
    ylabel = '[nT]'

  time_m, data_m = None, None

  if data_type == 'gic' and data_class == 'measured':
    time_m = data['modified']['time']
    data_m = data['modified']['data']
    time_m, data_m = subset(time_m, data_m, start, stop)
    legend = ['1-sec orig', '1-min avg']

  plt.figure()
  plot(time_o, data_o, title, ylabel, legend, time_m, data_m)
  savefig(sid, f'{base_name}', data_dir=data_dir)

  if data_type == 'mag' and data_class == 'measured':

    plt.figure()
    time_m = data['modified']['time']
    data_m = data['modified']['data']
    time_m, data_m = subset(time_m, data_m, start, stop)
    legend = mag_legend
    ylabel = '[nT]'
    title = f"{title} with mean removed"

    plot(time_m, data_m, title, ylabel, legend, None, None)
    savefig(sid, f'{base_name}-modified', data_dir=data_dir)

info, data_all = read(data_dir)

if sids is None:
  sids = info.keys()

for sid in sids: # site ids
  if sid not in info.keys():
    raise ValueError(f"Site '{sid}' not found in info.json")

  gic_types = info[sid]['GIC'].keys()
  if 'measured' and 'calculated' in gic_types:
    compare_gic(info, data_all, sid)

  # Plot original and modified data
  for data_type in info[sid].keys(): # e.g., GIC, B
    for data_class in info[sid][data_type].keys(): # e.g., measured, calculated
      data = data_all[sid][data_type][data_class]
      if isinstance(data, list):
        data_subclasses = info[sid][data_type][data_class]
        for idx, data_subclass in enumerate(data_subclasses):
          print(f"Plotting '{sid}/{data_type}/{data_class}/{data_subclass}'")
          plot_original(info, data[idx], sid, data_type, data_class, data_subclass)
      else:
        print(f"Plotting '{sid}/{data_type}/{data_class}'")
        plot_original(info, data, sid, data_type, data_class, None)

  # Special comparisons
  mag_types = info[sid]['B'].keys()
  if 'measured' in mag_types and 'calculated' in mag_types:
    compare_dB(info, data_all, sid)

  gic_types = info[sid]['GIC'].keys()
  if 'measured' and 'calculated' in gic_types:
    compare_gic(info, data_all, sid)

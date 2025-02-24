import os
import json
import numpy
import pickle
import shutil
import datetime
import time

import numpy as np
import pandas as pd
import numpy.ma as ma

from datetick import datetick

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-AGU-data')
all_dir  = os.path.join(data_dir, '_all')
all_file = os.path.join(all_dir, 'all.pkl')
base_dir = os.path.join(data_dir, '_processed')

plot_data = True    # Plot original and modified data
plot_compare = True # Plot measured and calculated data on same axes, when both available
sids = None # If none, plot all sites
# sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']
#sids = ['10052', '10064']

start = datetime.datetime(2024, 5, 10, 12, 0)
stop = datetime.datetime(2024, 5, 13, 0, 0)

def read(all_file, sid=None):
  fname = os.path.join('info', 'info.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    info_dict = json.load(f)

  info_df = pd.read_csv(os.path.join('info', 'info.csv'))

  fname = os.path.join('info', 'plot.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    plot_cfg = json.load(f)

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data, plot_cfg

def subset(time, data, start, stop):
  idx = numpy.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

def savefig(sid, fname, sub_dir="", fmts=['png']):
  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def compare_gic(info, data, sid):

  time_meas = data[sid]['GIC']['measured'][0]['modified']['time']
  data_meas = data[sid]['GIC']['measured'][0]['modified']['data']
  time_calc = data[sid]['GIC']['calculated'][0]['original']['time']
  data_calc = data[sid]['GIC']['calculated'][0]['original']['data']

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

  savefig(sid, 'GIC_compare_timeseries')

  plt.figure()
  cc = numpy.corrcoef(data_meas, data_calc)
  # Fixed calculation of pe
  numer = np.sum((data_meas-data_calc)**2)
  denom = np.sum((data_meas-data_meas.mean())**2)
  pe = ( 1-numer/denom )

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
  savefig(sid, 'GIC_compare_correlation')

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
  savefig(sid, 'GIC_histogram_meas_calc')

def compare_db(info, data, sid):

  time_meas = data[sid]['B']['measured'][0]['modified']['time']
  data_meas = data[sid]['B']['measured'][0]['modified']['data']
  time_meas, data_meas = subset(time_meas, data_meas, start, stop)
  data_meas = numpy.linalg.norm(data_meas, axis=1)

  model_names = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g']
  model_points = ['b.', 'g.']
  model_names = []
  for idx, data_source in enumerate(info[sid]['B']['calculated']):
    model_names.append(data_source.upper())
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
    label = model_names[idx]
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

  savefig(sid, 'B_compare_timeseries')
  
  # Plots to examine how well measured matched calculated values
  
  # Storage for correlation coefficient, prediction effeciency, and 
  # interpolated measured data
  cc = []
  pe = []
  data_interp = []

  # Plot illustrating correlation between measured and calculated
  plt.figure()
  plt.title(sid)
  
  # We have many more measured values than calculated values.
  # So we interpolate the measured to match up with the calculated values
  # To do so, we need to convert datetimes to timestamps so we can do interpolation
  time_meas_ts = [time.mktime(t.timetuple()) for t in time_meas]
  
  # Loop thru modeled (measured) results
  for idx in range(len(model_names)):
    time_calcs_ts = np.array( [time.mktime(t.timetuple()) for t in time_calcs[idx]] )
    
    # Get rid of the NaNs for cc and pe calculations
    time_calcs_ts = time_calcs_ts[~np.isnan(data_calcs[idx])]
    data_calcs[idx] = data_calcs[idx][~np.isnan(data_calcs[idx])]
    
    # Interpolate measured data
    data_interp.append( np.interp( time_calcs_ts, time_meas_ts, data_meas ) )
    
    # Add plot for each model
    label = model_names[idx].upper()
    plt.plot(data_interp[idx], data_calcs[idx], model_points[idx], markersize=1, label=label)
    
    cc.append( (np.corrcoef(data_interp[idx], data_calcs[idx]))[0,1] )
       
    numer = np.sum((data_interp[idx]-data_calcs[idx])**2)
    denom = np.sum((data_interp[idx]-data_interp[idx].mean())**2)
    pe.append( 1-numer/denom )

  # Write cc and pe on plot
  # TODO: Compute limits based on data
  text = f"{model_names[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}\n{model_names[1]} cc = {cc[1]:.2f} | pe = {pe[1]:.2f}"
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

  ylims = plt.gca().get_ylim()
  plt.plot([0, ylims[1]], [0, ylims[1]], color=3*[0.6], linewidth=0.5)

  plt.text(0.75*ylims[1], 100, text, **text_kwargs)
  plt.xlabel(r'Measured $\Delta B_H$ [nT]')
  plt.ylabel(r'Calculated $\Delta B_H$ [nT]')
  plt.grid()
  plt.xlim(ylims)
  plt.legend(loc='upper right')
  savefig(sid, 'B_compare_correlation')

  # Histograms showing delta between measured and calculated values
  plt.figure()
  
  # TODO: Compute binwidth from data
  # Setup bins
  bl = -1000
  bu = 1000
  bw = 50
  bins_c = numpy.arange(bl, bu+1, bw)
  bins_e = numpy.arange(bl-bw/2, bu+bw, bw)

  # Loop thru models and create data for histograms
  for idx in range(len(model_names)): 
    n_e, _ = numpy.histogram(data_interp[idx]-data_calcs[idx], bins=bins_e)
    plt.step(bins_c, n_e/sum(n_e), color=model_colors[idx], label=model_names[idx])
    
  # Add titles, legend, etc.
  plt.title(sid)
  # plt.xticks(bins_c[0::2])
  plt.xticks(fontsize=18)
  plt.xlabel(r'(Measured - Calculated) $\Delta B_H$ [nt]', fontsize=18)
  plt.xlim(bl-0.5, bu+0.5)
  plt.yticks(fontsize=18)
  plt.ylabel('Probability', fontsize=18)
  plt.grid(axis='y', color=[0.2,0.2,0.2], linewidth=0.2)
  plt.legend(loc='upper right')
  
  savefig(sid, 'B_histogram_meas_calc')


  # Plots to examine how well measured matched calculated values
  
  # Storage for correlation coefficient, prediction efficiency, and 
  # interpolated measured data
  cc = []
  pe = []
  data_interp = []

  # Plot illustrating correlation between measured and calculated
  plt.figure()
  plt.title(sid)
  
  # We have many more measured values than calculated values.
  # So we interpolate the measured to match up with the calculated values
  # To do so, we need to convert datetimes to timestamps so we can do interpolation
  time_meas_ts = [time.mktime(t.timetuple()) for t in time_meas]
  
  # Loop thru modeled (measured) results
  for idx in range(len(model_names)):
    time_calcs_ts = np.array( [time.mktime(t.timetuple()) for t in time_calcs[idx]] )
    
    # Get rid of the NaNs for cc and pe calculations
    time_calcs_ts = time_calcs_ts[~np.isnan(data_calcs[idx])]
    data_calcs[idx] = data_calcs[idx][~np.isnan(data_calcs[idx])]
    
    # Interpolate measured data
    data_interp.append( np.interp( time_calcs_ts, time_meas_ts, data_meas ) )
    
    # Add plot for each model
    label = model_names[idx].upper()
    plt.plot(data_interp[idx], data_calcs[idx], model_points[idx], markersize=1, label=label)
    
    cc.append( (np.corrcoef(data_interp[idx], data_calcs[idx]))[0,1] )
       
    numer = np.sum((data_interp[idx]-data_calcs[idx])**2)
    denom = np.sum((data_interp[idx]-data_interp[idx].mean())**2)
    pe.append( 1-numer/denom )

  # Write cc and pe on plot
  # TODO: Compute limits based on data
  text = f"{model_names[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}\n{model_names[1]} cc = {cc[1]:.2f} | pe = {pe[1]:.2f}"
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

  ylims = plt.gca().get_ylim()
  plt.plot([0, ylims[1]], [0, ylims[1]], color=3*[0.6], linewidth=0.5)

  plt.text(0.75*ylims[1], 100, text, **text_kwargs)
  plt.xlabel(r'Measured $\Delta B_H$ [nT]')
  plt.ylabel(r'Calculated $\Delta B_H$ [nT]')
  plt.grid()
  plt.xlim(ylims)
  plt.legend(loc='upper right')
  savefig(sid, 'B_compare_correlation')

  # Histograms showing delta between measured and calculated values
  plt.figure()
  
  # TODO: Compute binwidth from data
  # Setup bins
  bl = -1000
  bu = 1000
  bw = 50
  bins_c = numpy.arange(bl, bu+1, bw)
  bins_e = numpy.arange(bl-bw/2, bu+bw, bw)

  # Loop thru models and create data for histograms
  for idx in range(len(model_names)): 
    n_e, _ = numpy.histogram(data_interp[idx]-data_calcs[idx], bins=bins_e)
    plt.step(bins_c, n_e/sum(n_e), color=model_colors[idx], label=model_names[idx])
    
  # Add titles, legend, etc.
  plt.title(sid)
  # plt.xticks(bins_c[0::2])
  plt.xticks(fontsize=18)
  plt.xlabel(r'(Measured - Calculated) $\Delta B_H$ [nt]', fontsize=18)
  plt.xlim(bl-0.5, bu+0.5)
  plt.yticks(fontsize=18)
  plt.ylabel('Probability', fontsize=18)
  plt.grid(axis='y', color=[0.2,0.2,0.2], linewidth=0.2)
  plt.legend(loc='upper right')
  
  savefig(sid, 'B_histogram_meas_calc')

def plot_original(plot_info, data, sid, data_type, data_class, data_source, data_error):

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

  #print(f"  {sid}/{data_type}/{data_class}/{data_source}")
  title = f"{sid}/{data_type}/{data_class}"
  if data_error is not None:
    title += f"\n{data_error}"

  mag_legend = plot_info[data_source][data_type]
  sidx = sid.lower().replace(' ', '')
  base_name = f'{data_type}_{data_class}_{data_source}'

  # "o" for original.
  time_o = data['original']['time']
  data_o = data['original']['data']

  # Subset to desired time range
  time_o, data_o = subset(time_o, data_o, start, stop)

  ylabel = None
  if data_type == 'GIC':
    ylabel = 'GIC [A]'

  legend = None
  if data_type == 'B':
    legend = mag_legend
    ylabel = '[nT]'

  time_m, data_m = None, None

  if data_type == 'GIC' and data_class == 'measured':
    time_m = data['modified']['time']
    data_m = data['modified']['data']
    time_m, data_m = subset(time_m, data_m, start, stop)
    legend = ['1-sec orig', '1-min avg']

  plt.figure()
  plot(time_o, data_o, title, ylabel, legend, time_m, data_m)
  savefig(sid, f'{base_name}')

  if data_type == 'GIC' and data_class == 'measured':
    subdir = 'good' if data_error is None else 'bad'
    src_file = os.path.join(base_dir, sidx, f'{base_name}.png')
    dest_dir = os.path.join(all_dir, 'gic', subdir)
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)
    dest_file = os.path.join(dest_dir, f'{base_name}.png')
    print(f"  Copying\n    {src_file}\n    to\n    {dest_file}")
    shutil.copyfile(src_file, dest_file)

  if data_type == 'mag' and data_class == 'measured':

    plt.figure()
    time_m = data['modified']['time']
    data_m = data['modified']['data']
    time_m, data_m = subset(time_m, data_m, start, stop)
    legend = mag_legend
    ylabel = '[nT]'
    title = f"{title} with mean removed"

    plot(time_m, data_m, title, ylabel, legend, None, None)
    savefig(sid, f'{base_name}_modified')

  plt.close()


info_dict, info_df, data_all, plot_info = read(all_file)
if sids is None:
  sids = info_dict.keys()

if plot_data:

  for sid in sids: # site ids
    if sid not in info_dict.keys():
      raise ValueError(f"Site '{sid}' not found in info_dict.json")

    # Plot original and modified data
    for data_type in info_dict[sid].keys(): # e.g., GIC, B
      for data_class in info_dict[sid][data_type].keys(): # e.g., measured, calculated

        data = data_all[sid][data_type][data_class]
        data_sources = info_dict[sid][data_type][data_class]
        for idx, data_source in enumerate(data_sources):
          # Read info_df and row with site_id = sid, data_type = data_type,
          # data_class = data_class, data_source = data_source
          info_df_row = info_df[(info_df['site_id'] == sid) &
                                (info_df['data_type'] == data_type) &
                                (info_df['data_class'] == data_class) &
                                (info_df['data_source'] == data_source)]
          data_error = info_df_row['error'].values[0] if not info_df_row.empty else None
          data_error = str(data_error) if not pd.isnull(data_error) else None
          if data_error == 'nan':
            data_error = None

          if data[idx] is not None:
            print(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}'")
            plot_original(plot_info, data[idx], sid, data_type, data_class, data_source, data_error)
          else:
            print(f"  No data for '{sid}/{data_type}/{data_class}/{data_source}'")

    print(" ")

if plot_compare:

  for sid in sids: # site ids
    if sid not in info_dict.keys():
      raise ValueError(f"Site '{sid}' not found in info_dict.json")
      
    if 'B' in info_dict[sid].keys():
      mag_types = info_dict[sid]['B'].keys()
      if 'measured' in mag_types and 'calculated' in mag_types:
        print("  Plotting B measured and calculated")
        compare_db(info_dict, data_all, sid)

    if 'GIC' in info_dict[sid].keys():
      gic_types = info_dict[sid]['GIC'].keys()
      if 'measured' and 'calculated' in gic_types:
        print("  Plotting GIC measured and calculated")
        compare_gic(info_dict, data_all, sid)

###############################################################################################################

# comparison plots!

def savefig(fdir, fname, fmts=['png']):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        if fmt == 'png':
            plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def read_TVA_or_NERC(row):
  site_id = row['site_id']
  data_dir = os.path.join('..', '2024-AGU-data', 'processed')
  if row['data_source'] == 'NERC':
      #reading in data for site if NERC
      fname = os.path.join(data_dir, site_id, 'GIC_measured_NERC.pkl')
  elif row['data_source'] == 'TVA':
      #reading in data for site if TVA
      site_id = "".join(site_id.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, site_id, 'GIC_measured_TVA.pkl')

  with open(fname, 'rb') as f:
      site_data = pickle.load(f)

  site_df = pd.DataFrame(site_data)
  time = site_df['modified'][0]['time']
  mod_data = site_df['modified'][0]['data'] # 1-min avg data
  masked_data = ma.masked_invalid(mod_data) # 1-min data w nan values masked
  return time, mod_data, masked_data

sites = info_df['site_id'].tolist()

def compare_gic_site(sites):
  for idx_1, row in info_df.iterrows():

    site_1_id = row['site_id']
    if site_1_id not in sites:
      continue

    site_1_time, site_1_data, msk_site_1_data = read_TVA_or_NERC(row)

    for idx_2, row in info_df.iterrows():
      if idx_1 <= idx_2:  # Avoid duplicate pairs
        continue

      site_2_id = row['site_id']

      if site_2_id not in sites:
        continue

      site_2_time, site_2_data, msk_site_2_data = read_TVA_or_NERC(row)

      #plotting!!
      plt.figure()
      error_shift = 70
      yticks = np.arange(-120, 30, 10)
      labels = []
      for ytick in yticks:
          if ytick < -30:
              labels.append(str(ytick+error_shift))
          else:
              labels.append(str(ytick))
      kwargs = {"color": 'w', "linestyle": '-', "linewidth": 10, "xmin": 0, "xmax": 1}
      plt.axhline(y=-35, **kwargs)
      plt.axhline(y=-120, **kwargs)
      plt.title(f'{site_1_id} vs {site_2_id} GIC Comparison')
      plt.grid()
      plt.plot()
      plt.plot(site_1_time, site_1_data, label=site_1_id, linewidth=0.5)
      plt.plot(site_2_time, site_2_data, label=site_2_id, linewidth=0.5)
      plt.plot(site_1_time, site_1_data-site_2_data-error_shift, color=3*[0.3], label='difference', linewidth=0.5)
      plt.legend()
      plt.ylabel('[A]', rotation=0, labelpad=10)
      plt.ylim(-120, 30)
      plt.yticks(yticks, labels=labels)
      site_1_save =site_1_id.lower().replace(' ', '')
      site_2_save =site_2_id.lower().replace(' ', '')
      fname = f'{site_1_save}_{site_2_save}'
      out_dir = os.path.join('..', '2024-AGU-data', '_results', 'pairs')
      savefig(out_dir, fname)
      plt.close()

compare_gic_site(sites)
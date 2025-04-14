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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-May-Storm-data')
all_dir  = os.path.join(data_dir, '_all')
all_file = os.path.join(all_dir, 'all.pkl')
base_dir = os.path.join(data_dir, '_processed')

plot_data = False    # Plot original and modified data
plot_compare = True # Plot measured and calculated data on same axes, when both available
sids = None # If none, plot all sites
#sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']
#sids = ['10052', '10064']

# sids used to make plots for paper
paper_GIC_sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']
paper_B_sids = ['Bull Run', '50116']
sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union', '50116'] # Run for only paper sites

start = datetime.datetime(2024, 5, 10, 15, 0)
stop = datetime.datetime(2024, 5, 12, 6, 0)

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

def savefig(sid, fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join('..','2024-May-Storm-paper', sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def compare_gic(info, data, sid, save_hist=True):

  if 'modified' in data[sid]['GIC']['measured'][0]:
      time_meas = data[sid]['GIC']['measured'][0]['modified']['time']
      data_meas = data[sid]['GIC']['measured'][0]['modified']['data']
  else:
      time_meas = data[sid]['GIC']['measured'][0]['original']['time']
      data_meas = data[sid]['GIC']['measured'][0]['original']['data']
  time_meas, data_meas = subset(time_meas, data_meas, start, stop)

  model_names = []
  model_labels = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g']
  model_points = ['b.', 'g.']
  model_names = []
  for idx, data_source in enumerate(info[sid]['GIC']['calculated']):
    if 'nearest_sim_site' in data_source: #skip data_source dict
      continue
    model_names.append(data_source.upper())
    if data_source == 'TVA':
      time_calc = data[sid]['GIC']['calculated'][idx]['original']['time']
      data_calc = data[sid]['GIC']['calculated'][idx]['original']['data']
      time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])
      # TODO: Document why this is necessary
      data_calc = -data_calc
      model_labels.append(data_source.upper())

    if data_source == 'GMU':
      time_calc = data[sid]['GIC']['calculated'][idx]['original']['time']
      time_calc = np.array(time_calc).flatten()
      data_calc = data[sid]['GIC']['calculated'][idx]['original']['data'][:, 0:1]
      data_calc = np.array(data_calc).flatten()
      time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])

      cc = np.corrcoef(data_meas, data_calc)
      sim_site = info_dict[sid]['GIC']['calculated'][idx+1]['nearest_sim_site']

      if cc[0,1] < 0:
        data_calc = -data_calc
        model_labels.append(f'-{data_source.upper()}\n@ {sim_site}')
      else:
        model_labels.append(f'{data_source.upper()}\n@ {sim_site}')
    time_calcs.append(time_calc)
    data_calcs.append(data_calc)

  plt.figure()
  plt.title(sid)
  plt.grid()
  plt.plot()
  plt.plot(time_meas, data_meas, 'k', label='GIC Measured', linewidth=1)
  for idx in range(len(model_names)):
    label = model_labels[idx]
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)

  plt.legend()
  plt.ylabel('[A]', rotation=0, labelpad=10)
  plt.ylim(-50, 50)

  # add cc and pe to timeseries
  cc = []
  pe = []
  for idx in range(len(model_names)):
    cc.append(numpy.corrcoef(data_meas, data_calcs[idx])[0,1])
    # Fixed calculation of pe
    numer = np.sum((data_meas-data_calcs)**2)
    denom = np.sum((data_meas-data_meas.mean())**2)
    pe.append( 1-numer/denom )

  if len(model_names) == 1:
    text = f"{model_names[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}"
  elif len(model_names) == 2:
    text = f"{model_labels[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}\n{model_labels[1]} cc = {cc[1]:.2f} | pe = {pe[1]:.2f}"
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
  plt.text(time_calcs[0][0], 30, text, **text_kwargs)
  datetick()
  # get the legend object
  leg = plt.gca().legend()

  # change the line width for the legend
  for line in leg.get_lines():
      line.set_linewidth(1.5)

  savefig(sid, 'GIC_compare_timeseries')

  if sid in paper_GIC_sids:
    savefig_paper('GIC_compare_timeseries_NEW', sub_dir=f"{sid.lower().replace(' ', '')}")

  # Add the generated plot to the markdown file
  md_name = f"GIC_compare_timeseries.md"
  md_path = os.path.join(data_dir, md_name)
  with open(md_path, "a") as md_file:
    md_file.write(f"\n![](_processed/{sid.lower().replace(' ', '')}/GIC_compare_timeseries.png)\n")
  plt.close()

  for idx in range(len(model_names)):
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
    label = model_labels[idx]
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)
    plt.plot(time_calcs[idx], data_meas-data_calcs[idx]-error_shift, color=3*[0.3], label='Error', linewidth=0.5)

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

    savefig(sid, f'GIC_compare_timeseries_{model_names[idx]}')

    if sid in paper_GIC_sids:
      savefig_paper(f'GIC_compare_timeseries_{model_names[idx]}', sub_dir=f"{sid.lower().replace(' ', '')}")

    # Add the generated plot to the markdown file
    md_name = f"GIC_compare_timeseries_{model_names[idx]}.md"
    md_path = os.path.join(data_dir, md_name)
    with open(md_path, "a") as md_file:
      md_file.write(f"\n![](_processed/{sid.lower().replace(' ', '')}/GIC_compare_timeseries_{model_names[idx]}.png)\n")
    plt.close()

  
  plt.figure()
  cc = []
  pe = []
  for idx in range(len(model_names)):
    cc.append(numpy.corrcoef(data_meas, data_calcs[idx])[0,1])
    # Fixed calculation of pe
    numer = np.sum((data_meas-data_calcs[idx])**2)
    denom = np.sum((data_meas-data_meas.mean())**2)
    pe.append( 1-numer/denom )

    plt.plot(data_meas, data_calcs[idx], model_points[idx], markersize=1)

  if len(model_names) == 1:
    text = f"{model_names[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}"
  elif len(model_names) == 2:
    text = f"{model_labels[0]} cc = {cc[0]:.2f} | pe = {pe[0]:.2f}\n{model_labels[1]} cc = {cc[1]:.2f} | pe = {pe[1]:.2f}"
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
  # Set the aspect ratio to make the plot square and ensure xlim and ylim are the same
  ax = plt.gca()
  #ax.set_aspect('equal', adjustable='box')
  limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
  ax.set_xlim(limits)
  ax.set_ylim(limits)
  plt.plot([limits[0], limits[1]], [limits[0], limits[1]], color=3*[0.6], linewidth=0.5)
  plt.text(limits[0], limits[1], text, **text_kwargs)
  plt.xlabel('Measured GIC [A]')
  plt.ylabel('Calculated GIC [A]')
  plt.grid()
  savefig(sid, 'GIC_compare_correlation')
  if sid in paper_GIC_sids:
      savefig_paper(f'GIC_compare_correlation_NEW', sub_dir=f"{sid.lower().replace(' ', '')}")
  plt.close()

  # Reset the aspect ratio to normal
  ax.set_aspect('auto')

  # Add the generated plot to the markdown file
  md_name = f"GIC_compare_timeseries.md"
  md_path = os.path.join(data_dir, md_name)
  with open(md_path, "a") as md_file:
    md_file.write(f"\n![](_processed/{sid.lower().replace(' ', '')}/GIC_compare_correlation.png)\n")
  plt.close()

  plt.figure()
  #plt.title(name)
  bl = -10
  bu = 10
  bins_c = numpy.arange(bl, bu+1, 1)
  bins_e = numpy.arange(bl-0.5, bu+1, 1)
  for idx in range(len(model_names)): 
    n_e, _ = numpy.histogram(data_meas-data_calcs[idx], bins=bins_e)
    plt.step(bins_c, n_e/sum(n_e), color=model_colors[idx], label=model_labels[idx])
  plt.xticks(bins_c[0::2])
  plt.xticks(fontsize=18)
  plt.xlabel('(Measured - Calculated) GIC [A]', fontsize=18)
  plt.xlim(bl-0.5, bu+0.5)
  plt.yticks(fontsize=18)
  plt.ylabel('Probability', fontsize=18)
  plt.grid(axis='y', color=[0.2,0.2,0.2], linewidth=0.2)
  plt.legend(loc='upper right')
  savefig(sid, 'GIC_histogram_meas_calc')
  plt.close()

  if save_hist:
    # Add the generated plot to the markdown file
    md_name = f"GIC_compare_timeseries.md"
    md_path = os.path.join(data_dir, md_name)
    with open(md_path, "a") as md_file:
      md_file.write(f"\n![](_processed/{sid.lower().replace(' ', '')}/GIC_histogram_meas_calc.png)\n")
  plt.close()


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
  # create markdown files to hold plots
  if sids == info_dict.keys():
    markdown_files = [
            ("GIC_compare_timeseries.md", "GIC Compare Timeseries"),
            ("GIC_compare_timeseries_TVA.md", "GIC Compare Timeseries for TVA model"),
            ("GIC_compare_timeseries_GMU.md", "GIC Compare Timeseries for GMU simulation")
        ]

    for md_name, md_content in markdown_files:
      markdown_content = f"""# {md_content}"""
      md_path = os.path.join(data_dir, md_name)
      with open(md_path, "w") as md_file:
        md_file.write(markdown_content)
      print(f"Markdown file '{md_name}' created.")

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
        compare_gic(info_dict, data_all, sid, save_hist=False)

#########################################################################################################
stack_plot = False #make true to make stack plots :)

def plot_all_gic(info, info_df, data_all,  start, stop, data_source=['TVA', 'NERC'], offset=30):
    sids = info.keys()
    for source in data_source:
      print(f"Plotting {source} sites")
      source_sites = {'sites': [], 'lat': [], 'lon': []}
      for sid in sids:
          if 'GIC' in data_all[sid] and source in info[sid]['GIC']['measured']:
              site_info = info_df[
                  (info_df['site_id'] == sid) & 
                  (info_df['data_type'] == 'GIC') & 
                  (info_df['data_class'] == 'measured')
              ]
              if site_info['error'].isna().all():
                  source_sites['sites'].append(sid)
                  source_sites['lat'].append(site_info['geo_lat'].values[0])
                  source_sites['lon'].append(site_info['geo_lon'].values[0])
      # Sort sites by latitude
      sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
      source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)
      if len(sorted_sites) <= 20:
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
        subplots = False
      else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 11))
        subplots = True
      for i, sid in enumerate(source_sites['sites']):
          if 'GIC' in data_all[sid].keys() and source in info[sid]['GIC']['measured']:
              time = data_all[sid]['GIC']['measured'][0]['original']['time']
              data = data_all[sid]['GIC']['measured'][0]['original']['data']
              
              # Subset to desired time range
              time, data = subset(time, data, start, stop)

              if i < 20:
                ax_idx = 0
                # Add offset for each site
                data_with_offset = data + i * offset
              else:
                ax_idx = 1
                data_with_offset = data + (i-20) * offset

              
              # Plot the timeseries
              if subplots:
                  axes[ax_idx].plot(time, data_with_offset, linewidth=1)
              else:
                  axes.plot(time, data_with_offset, linewidth=1)
              # Add text to the plot to label waveform
              sid_lat = source_sites['lat'][i]
              sid_lon = source_sites['lon'][i]
              text = f'{sid} ({sid_lat:.1f},{sid_lon:.1f})'
              if subplots:
                axes[ax_idx].text(stop, data_with_offset[0]+5, text, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
              else:
                 axes.text(stop, data_with_offset[0]+5, text, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
      if subplots:
        for ax in axes:
            ax.grid()
            ax.yaxis.set_major_locator(plt.MultipleLocator(30))
            #ax.legend(loc='upper right')
            ax.yaxis.set_ticklabels([])  # Remove y-tick labels
            ax.set_ylim(max(axes[0].get_ylim(),axes[1].get_ylim()))
            datetick('x', axes=ax)
        fig.tight_layout()
      else:
        plt.grid()
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(30))
        #plt.legend(loc='upper right')
        plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
        datetick()
      
      # Save the figure
      savefig(f'_{source.lower()}', f'gic_{source.lower()}')
      savefig_paper(f'gic_{source.lower()}')
      plt.close()

# Call the function
if stack_plot:
  plot_all_gic(info_dict, info_df, data_all, start, stop)

###############################################################################################################
exit()
# comparison plots!

def savefig(fdir, fname, fmts=['png', 'pdf']):
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
  data_dir = os.path.join('..', '2024-May-Storm-data', 'processed')
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

#reading in info.extended.csv
fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
df = pd.read_csv(fname).set_index('site_id')
info_df = pd.read_csv(fname)

# Filter out sites with error message
info_df = info_df[~info_df['error'].str.contains('', na=False)]
# TODO: Print number of GIC sites removed due to error and how many kept.
# Remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)
sites = info_df['site_id'].tolist()

pkl_file = os.path.join('..', '2024-May-Storm-data', '_results', 'cc.pkl')
print(f"Reading {pkl_file}")
with open(pkl_file, 'rb') as file:
  cc_df = pickle.load(file)


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

      cc_row = cc_df[((cc_df['site_1'] == site_1_id) & (cc_df['site_2'] == site_2_id)) | 
          ((cc_df['site_2'] == site_1_id) & (cc_df['site_1'] == site_2_id))].iloc[0]
      cc = np.abs(cc_row['cc'])
      dist = cc_row['dist(km)']

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
      plt.title(f'{site_1_id} vs {site_2_id}\n|cc| = {cc:.2f}, distance = {dist:4.2f} km')
      plt.grid()
      plt.plot()
      if cc_row['cc'] < 0:
        site_1_data = -site_1_data
        plt.text(site_1_time[1500], -117, f'time series for {site_1_id} plotted inverse due to negative correlation', fontsize=8)
      plt.plot(site_1_time, site_1_data, label=site_1_id, linewidth=0.5)
      plt.plot(site_2_time, site_2_data, label=site_2_id, linewidth=0.5)
      plt.plot(site_1_time, site_1_data-site_2_data-error_shift, color=3*[0.3], label='difference', linewidth=0.5)
      plt.legend(loc='lower left')
      plt.ylabel('GIC [A]')
      plt.ylim(-120, 30)
      plt.yticks(yticks, labels=labels)
      datetick()
      site_1_save =site_1_id.lower().replace(' ', '')
      site_2_save =site_2_id.lower().replace(' ', '')
      fname = f'{site_1_save}_{site_2_save}'
      out_dir = os.path.join('..', '2024-May-Storm-data', '_results', 'pairs')
      savefig(out_dir, fname)
      plt.close()

compare_gic_site(sites)
import os
import json
import numpy
import pickle
import shutil
import datetime
import time

from swerve import DATA_DIR, plt_config, savefig, savefig_paper, subset, LOG_KWARGS, logger

import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt

from datetick import datetick

logger = logger(**LOG_KWARGS)

all_dir  = os.path.join(DATA_DIR, '_all')
all_file = os.path.join(all_dir, 'all.pkl')
base_dir = '_processed'
info_fname = os.path.join('info', 'info.extended.csv')
pkl_file = os.path.join(DATA_DIR, '_results', 'cc.pkl')

plot_data = False    # Plot original and modified data
plot_compare = True  # Plot measured and calculated data on same axes, when both available
stack_plot = False   # Plot GIC and dB_H stack plots
plot_pairs = False   # Plot and compare measured GIC across all "good" pairs
create_md = False    # updates md comparison files without replotting everything

paper = False        # only plots paper sites if true
sids = None         # If none, plot all sites; ignored if paper is True
if paper or sids is None:
  write_stats_df = True
else:
  write_stats_df = False
#sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']
#sids = ['10052', '10064']

# sids used to make plots for paper
paper_GIC_sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union']
paper_B_sids = ['Bull Run', '50116']
if paper:
  sids = ['Bull Run', 'Widows Creek', 'Montgomery', 'Union', '50116'] # Run for only paper sites

#sids = ['Bull Run']

limits = plt_config()

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


def add_subplot_label(ax, label, loc=(-0.15, 1)):
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


def format_cc_scatter(ax):
  # Sets the aspect ratio to make the plot square and ensure xlim and ylim are the same
  limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
  plt.plot([limits[0], limits[1]], [limits[0], limits[1]], color=3 * [0.6], linewidth=0.5)
  ticks = plt.xticks()[0]
  plt.xticks(ticks)
  plt.yticks(ticks)
  ax.set_xlim(limits)
  ax.set_ylim(limits)


def compare_gic(info, data, sid, show_sim_site=False, df=None):

  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''))

  if 'modified' in data[sid]['GIC']['measured'][0]:
    time_meas = data[sid]['GIC']['measured'][0]['modified']['time']
    data_meas = data[sid]['GIC']['measured'][0]['modified']['data']
  else:
    time_meas = data[sid]['GIC']['measured'][0]['original']['time']
    data_meas = data[sid]['GIC']['measured'][0]['original']['data']
  time_meas, data_meas = subset(time_meas, data_meas, limits['data'][0], limits['data'][1])

  model_names = []
  model_labels = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g']
  model_points = ['b.', 'g.']
  for idx, data_source in enumerate(info[sid]['GIC']['calculated']):
    if 'nearest_sim_site' in data_source: #skip data_source dict
      continue
    model_names.append(data_source.upper())
    if data_source == 'TVA':
      time_calc = data[sid]['GIC']['calculated'][idx]['original']['time']
      data_calc = data[sid]['GIC']['calculated'][idx]['original']['data']
      time_calc, data_calc = subset(time_calc, data_calc, limits['data'][0], time_meas[-1])
      # TODO: Document why this is necessary
      data_calc = -data_calc
      model_labels.append(data_source.upper())
    elif data_source == 'GMU':
      time_calc = data[sid]['GIC']['calculated'][idx]['original']['time']
      time_calc = np.array(time_calc).flatten()
      data_calc = data[sid]['GIC']['calculated'][idx]['original']['data'][:, 0:1]
      data_calc = np.array(data_calc).flatten()
      time_calc, data_calc = subset(time_calc, data_calc, limits['data'][0], time_meas[-1])

      sim_site = info_dict[sid]['GIC']['calculated'][idx+1]['nearest_sim_site']
      # cc will be calculated below, so just add label for now
      model_labels.append(f'Reference\n@ {sim_site}' if show_sim_site else f'Reference')
    time_calcs.append(time_calc)
    data_calcs.append(data_calc)

  # Calculate cc and pe once for all models
  cc = []
  pe = []
  for idx in range(len(model_names)):
    # for GMU, check sign and flip if needed
    if model_names[idx] == 'GMU':
      cc_val = np.corrcoef(data_meas, data_calcs[idx])[0, 1]
      if cc_val < 0:
        data_calcs[idx] = -data_calcs[idx]
        cc_val = np.corrcoef(data_meas, data_calcs[idx])[0, 1]
        if show_sim_site:
          sim_site = info_dict[sid]['GIC']['calculated'][idx+1]['nearest_sim_site']
          model_labels[idx] = fr'$-${model_labels[idx]}\n@ {sim_site}'
        else:
          model_labels[idx] = r'$-$' + model_labels[idx]
      cc.append(cc_val)
    else:
      cc.append(np.corrcoef(data_meas, data_calcs[idx])[0, 1])
    numer = np.sum((data_meas - data_calcs[idx]) ** 2)
    denom = np.sum((data_meas - data_meas.mean()) ** 2)
    pe.append(1 - numer / denom)


  # Plot all model timeseries on the same figure
  plt.figure()
  plt.title(sid)
  plt.grid()
  plt.plot()
  plt.plot(time_meas, data_meas, 'k', label='Measured', linewidth=1)
  for idx in range(len(model_names)):
    label = model_names[idx]
    if model_names[idx] == 'GMU':
      label = "Ref "
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)

  plt.ylabel('GIC [A]')
  plt.xlim(limits['xlims'][0], limits['xlims'][1])
  datetick()

  # get the legend object
  leg = plt.gca().legend()
  # change the line width for the legend
  for line in leg.get_lines():
    line.set_linewidth(1.5)

  savefig(fdir, 'GIC_compare_timeseries', logger)

  if sid in paper_GIC_sids:
    text = {
      'Bull Run': 'a)',
      'Montgomery': 'c)',
      'Union': 'e)',
      'Widows Creek': 'g)'
    }.get(sid, None)
    add_subplot_label(plt.gca(), text)
    savefig_paper(fdir, 'GIC_compare_timeseries', logger)
  plt.close()


  # Plot each model timeseries on its own figure
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

    plt.plot(time_meas, data_meas, 'k', label='Measured', linewidth=1)
    label = fr"{model_names[idx]} cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    if model_names[idx] == 'GMU':
      label = fr"Ref  cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)
    plt.plot(time_calcs[idx], data_meas - data_calcs[idx] - error_shift, color=3 * [0.3], label='Error', linewidth=0.5)
    plt.ylabel('GIC [A]')
    plt.yticks(yticks, labels=labels)
    plt.ylim(-80, 30)
    datetick()

    # get the legend object
    leg = plt.gca().legend(loc='upper right')
    # change the line width for the legend
    for line in leg.get_lines():
      line.set_linewidth(1.5)

    savefig(fdir, f'GIC_compare_timeseries_{model_names[idx]}', logger)
    plt.close()


  # Plot all model cc scatter on the same figure
  plt.figure()
  for idx in range(len(model_names)):
    if model_names[idx] == 'GMU':
      # Note: 2 Unicode thin spaces were inserted to pad Ref for alignment
      label = fr"Ref    cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    else:
      label = fr"{model_names[idx]} cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"

    plt.plot(data_meas, data_calcs[idx], model_points[idx], markersize=1, label=label)

  plt.title(sid)
  plt.xlabel('Measured GIC [A]')
  plt.ylabel('Calculated GIC [A]')
  plt.grid()

  # Set the aspect ratio to make the plot square and make {x,y}{lims,ticks} the same
  ax = plt.gca()
  lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
  plt.plot([lims[0], lims[1]], [lims[0], lims[1]], color=3 * [0.6], linewidth=0.5)
  ticks = plt.xticks()[0]
  plt.xticks(ticks)
  plt.yticks(ticks)
  ax.set_xlim(lims)
  ax.set_ylim(lims)

  # get the legend object
  leg = plt.gca().legend(loc='lower right', handletextpad=0.1)
  # change the marker size for the legend
  for line in leg.get_lines():
    line.set_markersize(6)

  savefig(fdir, 'GIC_compare_correlation', logger)

  if sid in paper_GIC_sids:
    text = {
      'Bull Run': 'b)',
      'Montgomery': 'd)',
      'Union': 'f)',
      'Widows Creek': 'h)'
    }.get(sid, None)
    add_subplot_label(plt.gca(), text)
    savefig_paper(fdir, 'GIC_compare_correlation', logger)

  plt.close()

  # Plot each model cc scatter on its own figure
  for idx in range(len(model_names)):
    plt.figure()
    plt.title(sid)
    if model_names[idx] == 'GMU':
      label = fr"GMU cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    else:
      label = fr"{model_names[idx]} cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    plt.plot(data_meas, data_calcs[idx], 'k.', markersize=1, label=label)
    ax = plt.gca()
    format_cc_scatter(ax)
    plt.xlabel('Measured GIC [A]')
    plt.ylabel('Calculated GIC [A]')
    plt.grid()
    savefig(fdir, f'GIC_compare_correlation_{model_names[idx]}', logger)
    plt.close()

  # Reset the aspect ratio to normal
  ax.set_aspect('auto')
  plt.close()

  # Histograms of error for all models on single figure
  plt.figure()
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
  savefig(fdir, 'GIC_histogram_meas_calc', logger)
  plt.close()

  if df is not None:
    if len(data_calcs) == 2:  # TVA and Ref
        df.loc[len(df)] = {
            'Site ID': sid,
            r'$\sigma$ [A]': f"{np.std(data_meas):.2f}",
            r'$\sigma_\text{TVA}$': f"{np.std(data_calcs[0]):.2f}",
            r'$\sigma_\text{Ref}$': f"{np.std(data_calcs[1]):.2f}",
            r'$\text{cc}^2_\text{TVA}$': f"{cc[0]**2:.2f}",
            r'$\text{cc}^2_\text{Ref}$': f"{cc[1]**2:.2f}",
            r'$\text{pe}_\text{TVA}$': f"{pe[0]:.2f}",
            r'$\text{pe}_\text{Ref}$': f"{pe[1]:.2f}"
        }
    else:
        df.loc[len(df)] = {
            'Site ID': sid,
            r'$\sigma$ [A]': f"{np.std(data_meas):.2f}",
            r'$\sigma_\text{TVA}$': None,
            r'$\sigma_\text{Ref}$': f"{np.std(data_calcs[0]):.2f}",
            r'$\text{cc}^2_\text{TVA}$': None,
            r'$\text{cc}^2_\text{Ref}$': f"{cc[0]**2:.2f}",
            r'$\text{pe}_\text{TVA}$': None,
            r'$\text{pe}_\text{Ref}$': f"{pe[0]:.2f}"
        }


def compare_db(info, data, sid, df=None):

  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''))

  time_meas = data[sid]['B']['measured'][0]['modified']['time']
  data_meas = data[sid]['B']['measured'][0]['modified']['data']
  time_meas, data_meas = subset(time_meas, data_meas, limits['data'][0], limits['data'][1])
  data_meas = numpy.linalg.norm(data_meas, axis=1)

  model_names = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g', 'orange']
  model_points = ['b.', 'g.', 'y.']
  model_names = []
  for idx, data_source in enumerate(info[sid]['B']['calculated']):
    model_names.append(data_source)
    time_calc = data[sid]['B']['calculated'][idx]['original']['time']
    data_calc = data[sid]['B']['calculated'][idx]['original']['data']
    time_calc, data_calc = subset(time_calc, data_calc, limits['data'][0], limits['data'][1])
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

  # get the legend object
  leg = plt.gca().legend()

  # change the line width for the legend
  for line in leg.get_lines():
      line.set_linewidth(1.5)

  savefig(fdir, 'B_compare_timeseries', logger)
  if sid in paper_B_sids:
    text = {
      'Bull Run': 'a)',
      '50116': 'c)',
    }.get(sid, None)
    add_subplot_label(plt.gca(), text)
    savefig_paper(fdir, 'B_compare_timeseries', logger)
  plt.close()

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

    cc.append( (np.corrcoef(data_interp[idx], data_calcs[idx]))[0,1] )

    numer = np.sum((data_interp[idx]-data_calcs[idx])**2)
    denom = np.sum((data_interp[idx]-data_interp[idx].mean())**2)
    pe.append( 1-numer/denom )

    # Add plot for each model
    label = fr"{model_names[idx]} cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    plt.plot(data_interp[idx], data_calcs[idx], marker='.', linestyle='None', color=model_colors[idx], markersize=1, label=label)

  ylims = plt.gca().get_ylim()
  plt.plot([0, ylims[1]], [0, ylims[1]], color=3*[0.6], linewidth=0.5)

  plt.xlabel(r'Measured $\Delta B_H$ [nT]')
  plt.ylabel(r'Calculated $\Delta B_H$ [nT]')
  plt.grid()
  format_cc_scatter(plt.gca())

  # get the legend object
  leg = plt.gca().legend(loc='upper right')
  # change the marker size for the legend
  for line in leg.get_lines():
      line.set_markersize(6)
  savefig(fdir, 'B_compare_correlation', logger)
  if sid in paper_B_sids:
    text = {
      'Bull Run': 'b)',
      '50116': 'd)',
    }.get(sid, None)
    add_subplot_label(plt.gca(), text)
    savefig_paper(fdir, 'B_compare_correlation', logger)
  plt.close()

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

  savefig(fdir, 'B_histogram_meas_calc', logger)

  if df is not None:
        df.loc[len(df)] = {
            'Site ID': sid,
            r'$\sigma$ [nT]': f"{np.std(data_meas):.1f}",
            r'$\sigma_\text{SWMF}$': f"{np.std(data_calcs[0]):.1f}",
            r'$\sigma_\text{MAGE}$': f"{np.std(data_calcs[1]):.1f}",
            r'$\sigma_\text{GGCM}$': f"{np.std(data_calcs[2]):.1f}",
            r'$\text{cc}^2_\text{SWMF}$': f"{cc[0]**2:.2f}",
            r'$\text{cc}^2_\text{MAGE}$': f"{cc[1]**2:.2f}",
            r'$\text{cc}^2_\text{GGCM}$': f"{cc[2]**2:.2f}",
            r'$\text{pe}_\text{SWMF}$': f"{pe[0]:.2f}",
            r'$\text{pe}_\text{MAGE}$': f"{pe[1]:.2f}",
            r'$\text{pe}_\text{GGCM}$': f"{pe[2]:.2f}"
        }


def plot_original(plot_info, data, sid, data_type, data_class, data_source, data_error):

  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''))
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

    plt.xlim(limits['xlim'][0], limits['xlim'][1])
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
  time_o, data_o = subset(time_o, data_o, limits['data'][0], limits['data'][1])

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
    time_m, data_m = subset(time_m, data_m, limits['data'][0], limits['data'][1])
    legend = ['1-sec orig', '1-min avg']

  plt.figure()
  plot(time_o, data_o, title, ylabel, legend, time_m, data_m)
  savefig(fdir, f'{base_name}', logger)

  if data_type == 'GIC' and data_class == 'measured':
    subdir = 'good' if data_error is None else 'bad'
    src_file = os.path.join(DATA_DIR, base_dir, sidx, f'{base_name}.png')
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
    time_m, data_m = subset(time_m, data_m, limits['data'][0], limits['data'][1])
    legend = mag_legend
    ylabel = '[nT]'
    title = f"{title} with mean removed"

    plot(time_m, data_m, title, ylabel, legend, None, None)
    savefig(fdir, f'{base_name}_modified', logger)

  plt.close()


def plot_all_gic(info, info_df, data_all, start, stop, data_source=['TVA', 'NERC'], offset=40):
    sids = info.keys()
    # note NERC sites that are TVA duplicates
    sid_copies = {'10197':'Sullivan',
                  '10204':'Shelby',
                  '10208':'Rutherford',
                  '10203':'Raccoon Mountain',
                  '10212':'Pinhook',
                  '10201':'Montgomery',
                  '10660':'Gleason',
                  '10200':'East Point',
                  '10207':'Bull Run'
                  }
    for source in data_source:
      print(f"Plotting {source} GIC sites")
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
                  source_sites['lat'].append(site_info['mag_lat'].values[0])
                  source_sites['lon'].append(site_info['mag_lon'].values[0])
      # Sort sites by latitude
      sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
      source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)

      if source == 'NERC':
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
      elif source == 'TVA':
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 5))

      offset_fix = 0
      for i, sid in enumerate(source_sites['sites']):
          if sid in sid_copies.keys():
            offset_fix +=1
            continue # skipping NERC sites that are TVA duplicates
          if 'GIC' in data_all[sid].keys() and source in info[sid]['GIC']['measured']:
              time = data_all[sid]['GIC']['measured'][0]['original']['time']
              data = data_all[sid]['GIC']['measured'][0]['original']['data']

              # Subset to desired time range
              time, data = subset(time, data, limits['data'][0], limits['data'][1])

              # Add offset for each site
              data_with_offset = data + (i * offset) - (offset_fix * offset)

              # Plot the timeseries
              axes.plot(time, data_with_offset, linewidth=0.5)
              # Add text to the plot to label waveform
              sid_lat = source_sites['lat'][i]
              sid_lon = source_sites['lon'][i]
              text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
              if sid in sid_copies.values():
                text = f'{sid}*\n({sid_lat:.1f},{sid_lon:.1f})'
              axes.text(datetime.datetime(2024, 5, 10, 11, 0), (i*offset)-(offset_fix*offset), text, fontsize=9, verticalalignment='center', horizontalalignment='left')
      plt.grid()
      plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
      #plt.legend(loc='upper right')
      plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
      plt.gca().set_xlim(limits['xlims'][0], limits['data'][1])
      plt.gca().set_ylim(-offset, max(data_with_offset)+10)

      axes.spines['top'].set_visible(False)
      axes.spines['right'].set_visible(False)
      axes.spines['left'].set_visible(False)
      #axes.spines['bottom'].set_position(('outward', 10))  # Adjust position of x-axis
      axes.yaxis.set_ticks_position('none')  # Remove y-axis ticks
      datetick()

      # remove first x gridline
      xgridlines = axes.get_xgridlines()
      gridline_of_interest = xgridlines[0]
      gridline_of_interest.set_visible(False)

      # Save the figure
      fdir = os.path.join(base_dir, f'_{source.lower()}')
      savefig(fdir, f'gic_{source.lower()}', logger)
      savefig_paper(fdir, f'gic_{source.lower()}', logger)
      plt.close()


def plot_all_db(info, info_df, data_all, start, stop,  offset=400):
  sids = info.keys()
  # note NERC sites that are TVA duplicates
  sid_copies = {
                  }

  print("Plotting all dB sites")
  source_sites = {'sites': [], 'lat': [], 'lon': []}
  for sid in sids:
    if 'B' in data_all[sid] and 'measured' in info[sid]['B']:
      site_info = info_df[
            (info_df['site_id'] == sid) & 
            (info_df['data_type'] == 'B') & 
            (info_df['data_class'] == 'measured')
        ]
      if site_info['error'].isna().all():
        source_sites['sites'].append(sid)
        source_sites['lat'].append(site_info['mag_lat'].values[0])
        source_sites['lon'].append(site_info['mag_lon'].values[0])
  # Sort sites by latitude
  sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
  source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)
  """
  # Loop over all pairs of sites and show sites w high correlation to determine duplicates
  for i, sid_1 in enumerate(source_sites['sites']):
    for sid_2 in source_sites['sites'][i+1:]:
      if 'B' in data_all[sid_1].keys() and 'B' in data_all[sid_2].keys():
        # Get data for both sites
        time1 = data_all[sid_1]['B']['measured'][0]['modified']['time']
        data1 = data_all[sid_1]['B']['measured'][0]['modified']['data']
        time1, data1 = subset(time1, data1, start, stop)
        data1 = numpy.linalg.norm(data1, axis=1)

        time2 = data_all[sid_2]['B']['measured'][0]['modified']['time']
        data2 = data_all[sid_2]['B']['measured'][0]['modified']['data']
        time2, data2 = subset(time2, data2, start, stop)
        data2 = numpy.linalg.norm(data2, axis=1)

        # Interpolate to common time base if needed
        if len(time1) != len(time2) or not np.all(time1 == time2):
          # Convert datetimes to timestamps for interpolation
          t1 = np.array([time.mktime(dt.timetuple()) for dt in time1])
          t2 = np.array([time.mktime(dt.timetuple()) for dt in time2])
          # Interpolate data2 to t1
          data2_interp = np.interp(t1, t2, data2)
          cc = np.corrcoef(data1, data2_interp)[0, 1]
        else:
          cc = np.corrcoef(data1, data2)[0, 1]

        if cc >= 0.995:
          text = f"cc={cc:.4f} between sites: {sid_1} and {sid_2}"
        else:
          text = None
        if text is not None:
          fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
          plt.plot(time1, data1, 'k', linewidth=0.5, label=sid_1)
          plt.plot(time2, data2, 'r', linewidth=0.5, label=sid_2)
          plt.legend()
          plt.text(datetime.datetime(2024, 5, 10, 11, 0), max(data1.max(), data2.max()), text, fontsize=9, verticalalignment='center', horizontalalignment='left')
          plt.show()
          plt.close()"""

  fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))

  offset_fix = 0

  for i, sid in enumerate(source_sites['sites']):
    if sid in sid_copies.keys():
      offset_fix +=1
      continue # skipping NERC sites that are TVA duplicates
    if 'B' in data_all[sid].keys():
      time_b = data_all[sid]['B']['measured'][0]['modified']['time']
      data = data_all[sid]['B']['measured'][0]['modified']['data']

      # Subset to desired time range
      time_b, data = subset(time_b, data, limits['data'][0], limits['data'][1])
      data = numpy.linalg.norm(data, axis=1)

      # Add offset for each site
      data_with_offset = data + (i*offset) - (offset_fix*offset)

      # Plot the timeseries
      axes.plot(time_b, data_with_offset, linewidth=0.5)
      # Add text to the plot to label waveform
      sid_lat = source_sites['lat'][i]
      sid_lon = source_sites['lon'][i]
      text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
      if sid in sid_copies.values():
        text = f'{sid}*\n({sid_lat:.1f},{sid_lon:.1f})'
      axes.text(limits['xlims'][0], (i*offset)-(offset_fix*offset), text, fontsize=9, verticalalignment='center', horizontalalignment='left')
  plt.grid()
  plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
  #plt.legend(loc='upper right')
  plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
  plt.gca().set_xlim(limits['xlims'][0], limits['data'][1])
  plt.gca().set_ylim(-offset, max(data_with_offset)+10)

  axes.spines['top'].set_visible(False)
  axes.spines['right'].set_visible(False)
  axes.spines['left'].set_visible(False)
  #axes.spines['bottom'].set_position(('outward', 10))  # Adjust position of x-axis
  axes.yaxis.set_ticks_position('none')  # Remove y-axis ticks
  datetick()

  # remove first x gridline
  xgridlines = axes.get_xgridlines()
  gridline_of_interest = xgridlines[0]
  gridline_of_interest.set_visible(False)

  # Save the figure
  fdir = os.path.join(base_dir, f'_db')
  savefig(fdir, 'db_all', logger)
  savefig_paper(fdir, 'db_all', logger)
  plt.close()


def gic_pairs(info, data, cc_df, sid_1, sid_2, lags):
  time_meas_1 = data[sid_1]['GIC']['measured'][0]['modified']['time']
  data_meas_1 = data[sid_1]['GIC']['measured'][0]['modified']['data']
  time_meas_1, data_meas_1 = subset(time_meas_1, data_meas_1, limits['data'][0], limits['data'][1])
  time_meas_2 = data[sid_2]['GIC']['measured'][0]['modified']['time']
  data_meas_2 = data[sid_2]['GIC']['measured'][0]['modified']['data']
  time_meas_2, data_meas_2 = subset(time_meas_2, data_meas_2, limits['data'][0], limits['data'][1])

  cc_row = cc_df[((cc_df['site_1'] == sid_1) & (cc_df['site_2'] == sid_2)) | 
          ((cc_df['site_2'] == sid_1) & (cc_df['site_1'] == sid_2))].iloc[0]
  cc = cc_row['cc']
  dist = cc_row['dist(km)']

  #plotting time series comparison
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
  plt.title(f'{sid_1} vs {sid_2}\n|cc| = {np.abs(cc):.2f}, distance = {dist:4.2f} km')
  plt.grid()
  plt.plot()
  if cc < 0:
    data_meas_1 = -data_meas_1
    plt.text(limits['data'][1], -117, f'time series for {sid_1} plotted inverse due to negative correlation', fontsize=8, 
         verticalalignment='bottom', horizontalalignment='right')
  plt.plot(time_meas_1, data_meas_1, 'b', label=sid_1, linewidth=0.5)
  plt.plot(time_meas_2, data_meas_2, 'r', label=sid_2, linewidth=0.5)
  plt.plot(time_meas_1, data_meas_1-data_meas_2-error_shift, color=3*[0.3], label='difference', linewidth=0.5)
  plt.legend(loc='lower left')
  plt.ylabel('GIC [A]')
  plt.ylim(-120, 30)
  plt.yticks(yticks, labels=labels)
  datetick()

  site_1_save =sid_1.lower().replace(' ', '')
  site_2_save =sid_2.lower().replace(' ', '')
  fname = f'{site_1_save}_{site_2_save}'
  out_dir = os.path.join('..', '2024-May-Storm-data', '_results', 'pairs')
  savefig(out_dir, fname)
  plt.close()


  #plotting cross and auto correlation
  plt.figure()

  cross_corr = [np.corrcoef(data_meas_1[~np.isnan(data_meas_1) & ~np.isnan(np.roll(data_meas_2, lag))], 
                            np.roll(data_meas_2, lag)[~np.isnan(data_meas_1) & ~np.isnan(np.roll(data_meas_2, lag))])[0, 1] for lag in lags]
  auto_corr_1 = [np.corrcoef(data_meas_1[~np.isnan(data_meas_1) & ~np.isnan(np.roll(data_meas_1, lag))], 
                             np.roll(data_meas_1, lag)[~np.isnan(data_meas_1) & ~np.isnan(np.roll(data_meas_1, lag))])[0, 1] for lag in lags]
  auto_corr_2 = [np.corrcoef(data_meas_2[~np.isnan(data_meas_2) & ~np.isnan(np.roll(data_meas_2, lag))], 
                             np.roll(data_meas_2, lag)[~np.isnan(data_meas_2) & ~np.isnan(np.roll(data_meas_2, lag))])[0, 1] for lag in lags]

  plt.plot(lags, cross_corr, 'k', label='Cross Corr')
  plt.plot(lags, auto_corr_1, 'b--', label=f'Auto Corr {sid_1}')
  plt.plot(lags, auto_corr_2, 'r--', label=f'Auto Corr {sid_2}')
  plt.xlabel('Lag [min]')
  plt.ylabel('Correlation')
  plt.title(f'{sid_1} vs {sid_2}')
  plt.legend(loc='upper left')
  plt.grid()

  fname = f'{site_1_save}_{site_2_save}_correlation'
  savefig(out_dir, fname)
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
            print(f"Plotting '{sid}/{data_type}/{data_class}/{data_source}'")
            plot_original(plot_info, data[idx], sid, data_type, data_class, data_source, data_error)
          else:
            print(f"  No data for '{sid}/{data_type}/{data_class}/{data_source}'")

    print(" ")

if plot_compare:
  if not write_stats_df:
    gic_df = None
    b_df = None
  else:
    gic_df = pd.DataFrame(columns=['Site ID', r'$\sigma$ [A]', r'$\sigma_\text{TVA}$', r'$\sigma_\text{Ref}$', 
                                   r'$\text{cc}^2_\text{TVA}$', r'$\text{cc}^2_\text{Ref}$',
                                   r'$\text{pe}_\text{TVA}$', r'$\text{pe}_\text{Ref}$'])
    b_df = pd.DataFrame(columns=['Site ID', r'$\sigma$ [nT]', r'$\sigma_\text{SWMF}$', r'$\sigma_\text{MAGE}$', 
                                   r'$\sigma_\text{GGCM}$', r'$\text{cc}^2_\text{SWMF}$', r'$\text{cc}^2_\text{MAGE}$',
                                   r'$\text{cc}^2_\text{GGCM}$', r'$\text{pe}_\text{SWMF}$', r'$\text{pe}_\text{MAGE}$',
                                   r'$\text{pe}_\text{GGCM}$'])

  for sid in sids: # site ids
    if sid not in info_dict.keys():
      raise ValueError(f"Site '{sid}' not found in info_dict.json")
    if 'B' in info_dict[sid].keys():
      mag_types = info_dict[sid]['B'].keys()
      if 'measured' in mag_types and 'calculated' in mag_types:
        print("Plotting B measured and calculated")
        compare_db(info_dict, data_all, sid, df=b_df)
    if 'GIC' in info_dict[sid].keys():
      gic_types = info_dict[sid]['GIC'].keys()
      if 'measured' and 'calculated' in gic_types:
        print("Plotting GIC measured and calculated")
        compare_gic(info_dict, data_all, sid, df=gic_df)
  if gic_df is not None:
    if paper:
      fname = os.path.join(DATA_DIR, "_results", "gic_table_paper")
    else:
      fname = os.path.join(DATA_DIR, "_results", "gic_table")
      gic_df.loc[len(gic_df)] = {
            'Site ID': 'Mean',
            r'$\sigma$ [A]': f"{pd.to_numeric(b_df[r'$\sigma$ [A]'], errors='coerce').mean():.1f}",
            r'$\sigma_\text{TVA}$': f"{pd.to_numeric(b_df[r'$\sigma_\text{TVA}$'], errors='coerce').mean():.1f}",
            r'$\sigma_\text{Ref}$': f"{pd.to_numeric(b_df[r'$\sigma_\text{Ref}$'], errors='coerce').mean():.1f}",
            r'$\text{cc}^2_\text{TVA}$': f"{pd.to_numeric(b_df[r'$\text{cc}^2_\text{TVA}$'], errors='coerce').mean():.2f}",
            r'$\text{cc}^2_\text{Ref}$': f"{pd.to_numeric(b_df[r'$\text{cc}^2_\text{Ref}$'], errors='coerce').mean():.2f}",
            r'$\text{pe}_\text{TVA}$': f"{pd.to_numeric(b_df[r'$\text{pe}_\text{TVA}$'], errors='coerce').mean():.2f}",
            r'$\text{pe}_\text{Ref}$': f"{pd.to_numeric(b_df[r'$\text{pe}_\text{Ref}$'], errors='coerce').mean():.2f}",
        }
    print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
    gic_df.to_markdown(fname + ".md", index=False, floatfmt=".2f")
    gic_df.to_latex(fname + ".tex", index=False, escape=False)
  if b_df is not None:
    if paper:
      fname = os.path.join(DATA_DIR, "_results", "b_table_paper")
    else:
      fname = os.path.join(DATA_DIR, "_results", "b_table")
      b_df.loc[len(b_df)] = {
            'Site ID': 'Mean',
            r'$\sigma$ [nT]': f"{pd.to_numeric(b_df[r'$\sigma$ [nT]'], errors='coerce').mean():.1f}",
            r'$\sigma_\text{SWMF}$': f"{pd.to_numeric(b_df[r'$\sigma_\text{SWMF}$'], errors='coerce').mean():.1f}",
            r'$\sigma_\text{MAGE}$': f"{pd.to_numeric(b_df[r'$\sigma_\text{MAGE}$'], errors='coerce').mean():.1f}",
            r'$\sigma_\text{GGCM}$': f"{pd.to_numeric(b_df[r'$\sigma_\text{GGCM}$'], errors='coerce').mean():.1f}",
            r'$\text{cc}^2_\text{SWMF}$': f"{pd.to_numeric(b_df[r'$\text{cc}^2_\text{SWMF}$'], errors='coerce').mean():.2f}",
            r'$\text{cc}^2_\text{MAGE}$': f"{pd.to_numeric(b_df[r'$\text{cc}^2_\text{MAGE}$'], errors='coerce').mean():.2f}",
            r'$\text{cc}^2_\text{GGCM}$': f"{pd.to_numeric(b_df[r'$\text{cc}^2_\text{GGCM}$'], errors='coerce').mean():.2f}",
            r'$\text{pe}_\text{SWMF}$': f"{pd.to_numeric(b_df[r'$\text{pe}_\text{SWMF}$'], errors='coerce').mean():.2f}",
            r'$\text{pe}_\text{MAGE}$': f"{pd.to_numeric(b_df[r'$\text{pe}_\text{MAGE}$'], errors='coerce').mean():.2f}",
            r'$\text{pe}_\text{GGCM}$': f"{pd.to_numeric(b_df[r'$\text{pe}_\text{GGCM}$'], errors='coerce').mean():.2f}"
        }
    print(f"Writing B prediction comparison tables to {fname}.{{md,tex}}")
    b_df.to_markdown(fname + ".md", index=False, floatfmt=".2f")
    b_df.to_latex(fname + ".tex", index=False, escape=False)


# create markdown files to hold plots
if create_md:
  markdown_files = [
            ("GIC_compare_timeseries.md", "GIC Compare Timeseries"),
            ("GIC_compare_timeseries_TVA.md", "GIC Compare Timeseries for TVA model"),
            ("GIC_compare_timeseries_GMU.md", "GIC Compare Timeseries for GMU simulation"),
            ("B_compare_timeseries.md", "B Compare Timeseries")
        ]

  for md_name, md_content in markdown_files:
    markdown_content = f"""# {md_content}"""
    md_path = os.path.join(DATA_DIR, md_name)
    with open(md_path, "w") as md_file:
      md_file.write(markdown_content)
    print(f"Markdown file '{md_name}' created.")

  model_names = ['TVA', 'GMU']

  for sid in info_dict.keys():
    if 'B' in info_dict[sid].keys():
      mag_types = info_dict[sid]['B'].keys()
      if 'measured' in mag_types and 'calculated' in mag_types:
        md_name = "B_compare_timeseries.md"
        md_path = os.path.join(DATA_DIR, md_name)
        img1 = os.path.join("_processed", sid.lower().replace(' ', ''), "B_compare_timeseries.png").replace("\\", "/")
        img2 = os.path.join("_processed", sid.lower().replace(' ', ''), "B_compare_correlation.png").replace("\\", "/")
        if os.path.exists(os.path.join(DATA_DIR, img1)):
          with open(md_path, "a") as md_file:
            md_file.write(f"\n![]({img1})\n")
        if os.path.exists(os.path.join(DATA_DIR, img2)):
          with open(md_path, "a") as md_file:
            md_file.write(f"\n![]({img2})\n")

    if 'GIC' in info_dict[sid].keys():
      gic_types = info_dict[sid]['GIC'].keys()
      if 'measured' and 'calculated' in gic_types:
        md_name = "GIC_compare_timeseries.md"
        md_path = os.path.join(DATA_DIR, md_name)
        img1 = os.path.join("_processed", sid.lower().replace(' ', ''), "GIC_compare_timeseries.png").replace("\\", "/")
        img2 = os.path.join("_processed", sid.lower().replace(' ', ''), "GIC_compare_correlation.png").replace("\\", "/")
        if os.path.exists(os.path.join(DATA_DIR, img1)):
          with open(md_path, "a") as md_file:
            md_file.write(f"\n![]({img1})\n")
        if os.path.exists(os.path.join(DATA_DIR, img2)):
          with open(md_path, "a") as md_file:
            md_file.write(f"\n![]({img2})\n")
        for idx, model_name in enumerate(model_names):
          md_name = f"GIC_compare_timeseries_{model_name}.md"
          md_path = os.path.join(DATA_DIR, md_name)
          img1 = os.path.join("_processed", sid.lower().replace(' ', ''), f"GIC_compare_timeseries_{model_name}.png").replace("\\", "/")
          img2 = os.path.join("_processed", sid.lower().replace(' ', ''), f"GIC_compare_correlation_{model_name}.png").replace("\\", "/")
          if os.path.exists(os.path.join(DATA_DIR, img1)):
              with open(md_path, "a") as md_file:
                  md_file.write(f"\n![]({img1})\n")
          if os.path.exists(os.path.join(DATA_DIR, img2)):
              with open(md_path, "a") as md_file:
                  md_file.write(f"\n![]({img2})\n")

# plotting stack plots for GIC and dB_H
if stack_plot:
  #reading in info.extended.csv
  print(f"Reading {info_fname}")
  info_df = pd.read_csv(info_fname)

  plot_all_gic(info_dict, info_df, data_all, limits['data'][0], limits['data'][1])
  plot_all_db(info_dict, info_df, data_all, limits['data'][0], limits['data'][1])


# comparison plots!
if plot_pairs:
  #reading in info.extended.csv
  print(f"Reading {info_fname}")
  info_df = pd.read_csv(info_fname)
  # Filter out sites with error message
  info_df = info_df[~info_df['error'].str.contains('', na=False)]
  # Remove rows that don't have data_type = GIC and data_class = measured
  info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
  info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
  info_df.reset_index(drop=True, inplace=True)
  good_sites = info_df['site_id'].tolist()

  print(f"Reading {pkl_file}")
  with open(pkl_file, 'rb') as file:
    cc_df = pickle.load(file)

  sids = good_sites
  lag = range(-60, 61, 1)
  for i, site_1 in enumerate(sids):
    for site_2 in sids[i+1:]:
      gic_pairs(info_dict, data_all, cc_df, site_1, site_2, lag)

  # make xcorr scatter plot
  lags = cc_df['peak_xcorr_lag(min)']
  xcorrs = cc_df['peak_xcorr']
  plt.figure()
  plt.scatter(lags, xcorrs, c='k')
  plt.xlabel('Lag [min]')
  plt.ylabel('Peak Cross Correlation')
  plt.grid()
  out_dir = os.path.join('..', '2024-May-Storm-data', '_results', 'pairs')
  savefig(out_dir, 'xcorr_scatter')
  plt.close()

  # make md
  markdown_files = [("GIC_compare_pairs.md", "GIC Compare Pairs")
          ]
  for md_name, md_content in markdown_files:
    markdown_content = f"""# {md_content}"""
    md_path = os.path.join(DATA_DIR, md_name)
    with open(md_path, "w") as md_file:
      md_file.write(markdown_content)
      md_file.write(f"\n![](_results/pairs/xcorr_scatter.png)\n")
    print(f"Writing markdown file: '{md_name}'.")
  for i, site_1 in enumerate(sids):
    for site_2 in sids[i+1:]:
      site_1_save =site_1.lower().replace(' ', '')
      site_2_save =site_2.lower().replace(' ', '')
      # Add the generated plot to the markdown file
      with open(md_path, "a") as md_file:
        md_file.write(f"\n![](_results/pairs/{site_1_save}_{site_2_save}.png)\n")
        md_file.write(f"\n![](_results/pairs/{site_1_save}_{site_2_save}_correlation.png)\n")
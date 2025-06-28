
def site_plot(sid, data, data_types=None, logger=None, show_plots=False):

  from swerve import config, savefig

  if logger is None:
    from swerve import LOG_KWARGS, logger
    logger = logger(**LOG_KWARGS)

  logger.info(f"Plotting '{sid}' data")

  CONFIG = config()
  out_dir = CONFIG['dirs']['processed']

  for data_type in data.keys(): # e.g., GIC, B

    if data_types is not None and data_type not in data_types:
      continue

    for data_class in data[data_type].keys(): # e.g., measured, calculated
      for data_source in data[data_type][data_class].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        if data[data_type][data_class][data_source] is not None:
          logger.info(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}'")
          _plot(data[data_type][data_class][data_source], show_plots=show_plots)
          base_dir = f"{out_dir}/{sid.lower().replace(' ', '')}/data/figures"
          savefig(base_dir, f"{data_type}_{data_class}_{data_source}", logger=logger, logger_indent=4)
        else:
          logger.info(f"  No data for '{sid}/{data_type}/{data_class}/{data_source}'")

    if 'measured' in data[data_type] and 'calculated' in data[data_type]:
      logger.info(f"  Plotting measured/calculated comparison for '{sid}/{data_type}'")
      _plot_compare(sid, data_type, data[data_type], logger)


def _plot(data, show_plots=False):
  from matplotlib import pyplot as plt
  from datetick import datetick

  from swerve import plt_config

  plt.close()
  plt_config()

  component_labels = data['original']['label']
  if isinstance(component_labels, str):
    component_labels = [component_labels]

  labels_orig = []
  labels_mod = []
  for component_label in component_labels:
    labels_orig.append(f"{component_label} [{data['original']['units']}] Original")
    labels_mod.append(f"{component_label} [{data['original']['units']}] Modified")

  errors = []
  if 'error' in data:
    errors = [f"Original Error: {data['error']}"]

  plt.plot(data['original']['time'], data['original']['data'], label=labels_orig, color='blue', lw=2)
  if 'modified' in data and 'error' not in data:
    if 'error' in data['modified']:
      errors.append(f"Modified Error: {data['modified']['error']}")
    if 'time' in data['modified'] and 'data' in data['modified']:
      plt.plot(data['modified']['time'], data['modified']['data'], label=labels_mod, color='orange', lw=1)
    errors = '\n'.join(errors)
    plt.title(f"{errors}Modification = {data['modified']['modification']}")
  else:
    plt.title("\n".join(errors))

  plt.legend()
  plt.grid(True)
  datetick('x')
  if show_plots:
    plt.show()


def _plot_compare(sid, data_type, data, logger):

  import os
  import numpy

  from matplotlib import pyplot as plt
  from datetick import datetick

  from swerve import config, savefig, savefig_paper, add_subplot_label

  CONFIG = config()
  out_dir = CONFIG['dirs']['processed']

  def _savefig_paper(sid, data_type, ftype, fdir, fname):
    if sid in CONFIG['paper_sids'][data_type]['correlation'].keys():
      text = CONFIG['paper_sids'][data_type]['correlation'][sid]
      add_subplot_label(plt.gca(), text)
      savefig_paper(fdir, fname, logger, logger_indent=4)

  def format_cc_scatter(ax):
    # Sets the aspect ratio to make the plot square and ensure xlim and ylim are the same
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    plt.plot([limits[0], limits[1]], [limits[0], limits[1]], color=3 * [0.6], linewidth=0.5)
    ticks = plt.xticks()[0]
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

  fdir = os.path.join(out_dir, sid.lower().replace(' ', ''), 'figures')

  m_keys = data['measured'].keys()
  if len(m_keys) > 1:
    logger.warning(f"    Multiple measured data sources for {sid}: {m_keys}. Using the first one.")
  m_key = list(m_keys)[0]
  if 'error' in data['measured'][m_key]['modified']:
    msg = "    Skipping comparison b/c no modified measured data for "
    msg += f"{sid}/{data_type}/{m_key} due to error: {data['measured'][m_key]['modified']['error']}"
    logger.warning(msg)
    return

  time_meas = data['measured'][m_key]['modified']['time']
  data_meas = data['measured'][m_key]['modified']['data']

  if data_type == 'B':
    # Compute dB_H (All data has had baseline removed, so data_type is dB.)
    data_meas = numpy.linalg.norm(data_meas[:, 0:2], axis=1)

  model_names = []
  time_calcs = []
  data_calcs = []
  model_colors = ['b', 'g', 'y']
  model_points = ['b.', 'g.', 'y.']
  model_names = []
  for data_source in data['calculated'].keys():
    model_names.append(data_source)

    time_calc = data['calculated'][data_source]['modified']['time']
    time_calcs.append(time_calc)

    data_calc = data['calculated'][data_source]['modified']['data']
    if data_type == 'B':
      data_calc = numpy.linalg.norm(data_calc[:, 0:2], axis=1)
    data_calcs.append(data_calc)

  plt.figure()
  plt.title(sid)
  plt.plot(time_meas, data_meas, 'k', linewidth=1, label='Measured')
  for idx in range(len(model_names)):
    label = model_names[idx]
    plt.plot(time_calcs[idx], data_calcs[idx], model_colors[idx], linewidth=0.4, label=label)
  if data_type == 'B':
    plt.ylabel(r'$\Delta B_H$ [nT]')
  datetick()
  plt.legend()
  plt.grid()

  # get the legend object
  leg = plt.gca().legend()

  # change the line width for the legend
  for line in leg.get_lines():
      line.set_linewidth(1.5)

  fname = f'{data_type}_compare_timeseries'
  savefig(fdir, fname, logger, logger_indent=4)
  _savefig_paper(sid, data_type, 'timeseries', fdir, fname)
  plt.close()

  # Correlation between measured and calculated
  plt.figure()
  plt.title(sid)

  # Loop thru modeled (measured) results
  for idx in range(len(model_names)):
    #data_calcs[idx] = data_calcs[idx][~np.isnan(data_calcs[idx])]
    # Add plot for each model
    #label = fr"{model_names[idx]} cc$^2$ = {cc[idx]**2:.2f} | pe = {pe[idx]:.2f}"
    label = fr"{model_names[idx]}"
    plt.plot(data_meas, data_calcs[idx], model_points[idx], markersize=1, label=label)

  ylims = plt.gca().get_ylim()
  plt.plot([0, ylims[1]], [0, ylims[1]], color=3*[0.6], linewidth=0.5)

  if data_type == 'B':
    plt.xlabel(r'Measured $\Delta B_H$ [nT]')
    plt.ylabel(r'Calculated $\Delta B_H$ [nT]')
  plt.grid()
  format_cc_scatter(plt.gca())

  # get the legend object
  leg = plt.gca().legend(loc='upper right')
  # change the marker size for the legend
  for line in leg.get_lines():
      line.set_markersize(6)

  fname = f'{data_type}_compare_correlation'
  savefig(fdir, fname, logger, logger_indent=4)
  _savefig_paper(sid, data_type, 'correlation', fdir, fname)
  plt.close()

  return
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

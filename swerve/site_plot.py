
def site_plot(sid, data, data_types=None, logger=None, show_plots=False):

  import os

  from swerve import config, savefig

  if logger is None:
    from swerve import LOG_KWARGS, logger
    logger = logger(**LOG_KWARGS)

  logger.info(f"Plotting '{sid}' data")

  CONFIG = config()
  out_dir = CONFIG['dirs']['processed']

  for data_type in data.keys(): # e.g., GIC, B

    if data_types is not None and data_type not in data_types:
      # Skip this data type if not in requested data_types to plot.
      logger.info(f"  Not plotting '{sid}/{data_type}' data type b/c not in requested data_types = {data_types}.")
      continue

    base_dir = f"{out_dir}/{sid.lower().replace(' ', '')}/figures"
    dir_original = os.path.join(base_dir, 'original')
    dir_compare = os.path.join(base_dir, 'compare')

    # Plot measured vs calculated data
    if 'measured' in data[data_type] and 'calculated' in data[data_type]:
      for calculated_source in data[data_type]['calculated'].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        for style in ['timeseries', 'scatter']:
          logger.info(f"  Plotting '{sid}/{data_type}/{calculated_source}' vs. measured data as {style}")
          _plot_measured_vs_calculated(data[data_type], calculated_source, sid, style=style, show_plots=show_plots)
          fname = f"{data_type}_calculated_{calculated_source}_vs_measured_{style}"
          savefig(dir_compare, fname, logger=logger, logger_indent=4)

    # Plot original vs modified data
    for data_class in data[data_type].keys(): # e.g., measured, calculated
      for data_source in data[data_type][data_class].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        if data[data_type][data_class][data_source] is not None:
          logger.info(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}' original vs. modified data")
          _plot_measured_original_vs_modified(data[data_type][data_class][data_source], sid, show_plots=show_plots)
          fname = f"{data_type}_{data_class}_{data_source}"
          savefig(dir_original, fname, logger=logger, logger_indent=4)
        else:
          logger.info(f"  No data for '{sid}/{data_type}/{data_class}/{data_source}'")

def _plot_measured_vs_calculated(data, calculated_source, sid, style='timeseries', show_plots=False):

  measured_source = data['measured'].keys()
  #if len(m_keys) > 1:
  #  logger.warning(f"    Multiple measured data sources for {sid}: {measured_source}. Using the first one.")
  measured_source = list(measured_source)[0]
  measured_modified = data['measured'][measured_source]['modified']

  calculated = data['calculated'][calculated_source]['modified']
  calculated_metrics = data['calculated'][calculated_source]['modified']['metrics']

  unit = data['measured'][measured_source]['original']['unit']
  component_labels1 = data['measured'][measured_source]['original']['labels'].copy()
  for idx, label in enumerate(component_labels1):
    if style == 'scatter':
      component_labels1[idx] = f"{label} {measured_source} [{unit}]"
    if style == 'timeseries':
      component_labels1[idx] = f"{label} {measured_source}"

  component_labels2 = data['calculated'][calculated_source]['original']['labels'].copy()
  texts = None
  if style == 'scatter':
    texts = []
  for idx, label in enumerate(component_labels2):
    cc = f"{calculated_metrics['cc'][idx]:.2f}"
    pe = f"{calculated_metrics['pe'][idx]:.2f}"
    metrics = f"cc = ${cc}$ | pe = ${pe}$"
    if style == 'scatter':
      component_labels2[idx] = f"{label} {calculated_source} [{unit}]"
      texts.append(metrics)
    if style == 'timeseries':
      component_labels2[idx] = f"{label} {calculated_source} {metrics}"

  if style == 'scatter':
    ylabels = None

  if style == 'timeseries':
    ylabels = len(component_labels2)*[f"[{unit}]"]

  kwargs = {
    'ylabels': ylabels,
    'component_labels1': component_labels1,
    'component_labels2': component_labels2,
    'texts': texts,
    'suptitle': sid,
    'style': style,
    'show_plots': show_plots
  }

  _plot_stack(measured_modified, calculated, **kwargs)


def _plot_measured_original_vs_modified(data, sid, show_plots=False):

  component_labels1 = data['original']['labels'].copy()
  for idx, label in enumerate(component_labels1):
    component_labels1[idx] = f"{label} original"

  component_labels2 = data['modified']['labels'].copy()
  for idx, label in enumerate(component_labels2):
    component_labels2[idx] = f"{label} modified"

  kwargs = {
    'ylabels': len(component_labels2)*[f"[{data['original']['unit']}]"],
    'component_labels1': component_labels1,
    'component_labels2': component_labels2,
    'style': 'timeseries',
    'suptitle': sid,
    'show_plots': show_plots
  }

  original = data['original']
  modified = data['modified']

  if 'error' in data['original']:
    kwargs['suptitle'] = f"Original Error: {data['original']['error']}"
    _plot_stack(None, None, 'original', 'modified', **kwargs)
  else:
    kwargs['suptitle'] = f"Modification = {data['modified']['modification']}"
    _plot_stack(original, modified, **kwargs)


def _plot_stack(data1, data2, ylabels, component_labels1, component_labels2, texts=None, suptitle=None, style='timeseries', show_plots=False):

  from matplotlib import pyplot as plt
  from datetick import datetick

  from swerve import plt_config, format_cc_scatter

  plt.close()
  if style == 'scatter':
    plt_config(scale=0.5)
  if style == 'timeseries':
    plt_config(scale=0.75)
  plt.figure()

  line1_opts = {'color': 'blue', 'lw': 2}
  line2_opts = {'color': 'orange', 'lw': 1}
  sharex = False
  if style == 'timeseries':
    sharex = True

  n_stack = data1['data'].shape[1]
  gs = plt.gcf().add_gridspec(n_stack)
  axes = gs.subplots(sharex=sharex)
  if n_stack == 1:
    axes = [axes]

  if data1 is None:
    return

  for j in range(n_stack):
    if j == 0 and suptitle is not None:
      # Use title() for force suptitle to be centered on axis
      axes[j].set_title(suptitle)

    if ylabels is not None:
      axes[j].set_ylabel(ylabels[j])

    if style == 'timeseries':
      axes[j].plot(data1['time'], data1['data'][:, j], label=component_labels1[j], **line1_opts)
      axes[j].plot(data2['time'], data2['data'][:, j], label=component_labels2[j], **line2_opts)
      axes[j].legend(ncol=2, frameon=True, loc='upper right')

    if style == 'scatter':
      axes[j].scatter(data1['data'][:, j], data2['data'][:, j], label=component_labels2[j], s=1, color='black')
      axes[j].set_xlabel(component_labels1[j])
      axes[j].set_ylabel(component_labels2[j])
      format_cc_scatter(axes[j])
      bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", linewidth=0.8, alpha=0.5)
      axes[j].text(0.02, 0.98, texts[j], transform=axes[j].transAxes,
             fontsize=6, verticalalignment='top', horizontalalignment='left',
             bbox=bbox_props)

    axes[j].grid(True)

  plt.gcf().align_ylabels(axes)

  if style == 'timeseries':
    datetick('x')

  if show_plots:
    plt.show()


def site_plot(sid, data, data_types=None, logger=None, show_plots=False):

  import os

  from swerve import config, savefig, savefig_paper

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

    base_dir = f"{out_dir}/sites/{sid.lower().replace(' ', '')}/figures"
    dir_raw = os.path.join(base_dir, 'raw')
    dir_original = os.path.join(base_dir, 'original')
    dir_compare = os.path.join(base_dir, 'compare')

    # If data_raw is available, plot it
    for data_class in data[data_type].keys():
      for data_source in data[data_type][data_class].keys():
        if 'data_raw' in data[data_type][data_class][data_source]['original']:
          logger.info(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}/raw' data")
          _plot_raw(data[data_type][data_class][data_source]['original'], sid, show_plots=show_plots)
          fname = f"{data_type}_{data_class}_{data_source}"
          savefig(dir_raw, fname, logger=logger, logger_indent=4)
          if data_source == 'MAGE':
            # Special case for MAGE data, plot raw data in a different way
            logger.info(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}/raw' to compare dBn with dBt and dBp")
            _plot_raw_mage(data[data_type][data_class][data_source]['original'], sid, show_plots=show_plots)
            fname = f"{data_type}_{data_class}_{data_source}_compare_dBn_dBt_dBp"
            savefig(dir_raw, fname, logger=logger, logger_indent=4)


    # Plot measured vs calculated data
    if 'measured' in data[data_type] and 'calculated' in data[data_type]:
      for style in ['timeseries', 'scatter']:
        if 'paper_sids' in CONFIG.keys() and sid in CONFIG['paper_sids'][data_type][style]:
          paper_dir = os.path.join(CONFIG['dirs']['paper'], 'figures', '_processed', f'{sid.lower().replace(' ', '')}')
          subplot_label = CONFIG['paper_sids'][data_type][style][sid]
        else: subplot_label = None
        if len(data[data_type]['calculated'].keys()) > 1: # if multiple calculated sources, plot all vs measured
          logger.info(f"  Plotting all '{sid}/{data_type}' calculated vs. measured data as {style}")
          plots = _plot_measured_vs_calculated(data[data_type], None, sid, style=style, subplot_label=subplot_label, show_plots=show_plots)
          fname = f"_calculated_all_vs_measured_{style}"
          _save_plots(plots, fname, dir_compare, logger=logger)
          if subplot_label != None:
            fname = f"{data_type}_compare_{style}"
            savefig_paper(paper_dir, fname, logger=logger, logger_indent=4)
        if len(data[data_type]['calculated'].keys()) > 1: # if multiple calculated sources, plot all vs measured
          logger.info(f"  Plotting all '{sid}/{data_type}' calculated vs. measured data as {style}")
          plots = _plot_measured_vs_calculated(data[data_type], None, sid, style=style, subplot_label=subplot_label, show_plots=show_plots)
          fname = f"_calculated_all_vs_measured_{style}"
          _save_plots(plots, fname, dir_compare, logger=logger)
          if subplot_label != None:
            fname = f"{data_type}_compare_{style}"
            savefig_paper(paper_dir, fname, logger=logger, logger_indent=4)
        for calculated_source in data[data_type]['calculated'].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
          logger.info(f"  Plotting '{sid}/{data_type}/{calculated_source}' vs. measured data as {style}")
          plots = _plot_measured_vs_calculated(data[data_type], calculated_source, sid, style=style, show_plots=show_plots)
          fname = f"_calculated_{calculated_source}_vs_measured_{style}"
          _save_plots(plots, fname, dir_compare, logger=logger)

    # Plot original vs modified data
    for data_class in data[data_type].keys(): # e.g., measured, calculated
      for data_source in data[data_type][data_class].keys(): # e.g., TVA, NERC, SWMF, OpenGGCM
        if data[data_type][data_class][data_source] is not None:
          logger.info(f"  Plotting '{sid}/{data_type}/{data_class}/{data_source}' original vs. modified data")
          plots = _plot_measured_original_vs_modified(data[data_type][data_class][data_source], sid, show_plots=show_plots)
          fname = f"{data_type}_{data_class}_{data_source}"
          _save_plots(plots, fname, dir_original, logger=logger, include_label=False)
        else:
          logger.info(f"  No data for '{sid}/{data_type}/{data_class}/{data_source}'")


def _plot_raw_mage(data, sid, show_plots=False):
  # Special case for MAGE data, plot raw data in a different way
  from matplotlib import pyplot as plt
  from datetick import datetick
  from swerve import plt_config
  plt.close()
  gs = plt.gcf().add_gridspec(2)
  axes = gs.subplots(sharex=True)

  ylabels = data['labels_raw'].copy()
  plt_config(scale=0.75)
  import numpy as np
  if sid == 'Bull Run':
    # Bull Run data has a different declination
    decl = np.deg2rad(2.3)
  elif sid == 'Union':
    # Union data has a different declination
    decl = np.deg2rad(5.9)
  else:
    decl = np.deg2rad(0) #TODO; specify or other sites?
  #dBn_mag = -dBt_geo*cos(decl) + dBp_geo*sin(decl)
  # using dBn_geo = -dBt_geo
  #dBn_mag =  dBn_geo*cos(decl) + dBp_geo*sin(decl)
  # z in mag aligns with z in geo rotated towards geographic east by declination angle D
  dBn = -data['data_raw'][:,1]*np.cos(decl) + data['data_raw'][:,2]*np.sin(decl)
  dBn = -dBn # Why?
  label_dBn = fr"dBn = dBt$\cdot \cos(D)$ $-$ dBp$\cdot \sin(D)$; ($D$ = decl. = {np.rad2deg(decl):.1f}$^\circ$)"

  axes[0].plot(data['time'], data['data_raw'][:,0], label=f"{ylabels[0]} 1st val = ${data['data_raw'][0,0]:.4f}$", color='blue', lw=2)
  axes[0].plot(data['time'], data['data_raw'][:,1], label=f"{ylabels[1]} 1st val = ${data['data_raw'][0,1]:.4f}$", color='red', lw=1)
  #axes[0].plot(data['time'], dBn, label=label_dBn, color='black', lw=1)
  axes[0].grid(True)
  axes[0].set_title(sid)
  axes[0].legend()

  axes[1].plot(data['time'],  data['data_raw'][:,0], label=f"{ylabels[0]} 1st val = ${data['data_raw'][0,0]:.4f}$", color='blue', lw=2)
  axes[1].plot(data['time'], -data['data_raw'][:,2], label=f"-{ylabels[2]} 1st val = $-1\\cdot ({data['data_raw'][0,2]:.4f})$", color='red', lw=1)

  axes[1].grid(True)
  axes[1].set_title(sid)
  axes[1].legend()
  datetick('x')


def _plot_raw(data, sid, show_plots=False):
  data1 = {"time": data['time'], "data": data['data_raw']}
  ylabels = data['labels_raw'].copy()
  _plot_stack(data1, None, ylabels, None, None, suptitle=sid, style='timeseries', show_plots=show_plots)


def _plot_measured_vs_calculated(data, calculated_source, sid, style='timeseries', subplot_label=None, show_plots=False):

  measured_source = data['measured'].keys()
  #if len(m_keys) > 1:
  #  logger.warning(f"    Multiple measured data sources for {sid}: {measured_source}. Using the first one.")
  measured_source = list(measured_source)[0]
  measured_modified = data['measured'][measured_source]['modified']

  unit = data['measured'][measured_source]['original']['unit']

  if calculated_source is None:
    calculated_sources = data['calculated'].keys()
  else:
    calculated_sources = calculated_source.split(',')

  ylabels = []
  component_labels1 = data['measured'][measured_source]['original']['labels'].copy()
  for idx, label in enumerate(component_labels1):
    if label =='B_H': label = '$\\Delta B_H$'
    if style == 'scatter':
      if len(calculated_sources) == 1:
        component_labels1[idx] = f"Measured {label} {measured_source} [{unit}]"
      else:
        component_labels1[idx] = f"Measured {label} [{unit}]"
    if style == 'timeseries':
      ylabels.append(f"{label} [{unit}]")
      component_labels1[idx] = f"Measured"
  
  calculated = {}
  calculated_metrics = {}
  component_labels2 = {}
  fit = {}
  for source in calculated_sources:
    calculated[source] = data['calculated'][source]['modified']
    calculated_metrics[source] = data['calculated'][source]['modified']['metrics']
    component_labels2[source] = data['calculated'][source]['original']['labels'].copy()
    fit[source] = False

    for idx, label in enumerate(component_labels2[source]):
      cc = calculated_metrics[source]['cc'][idx]
      cc2 = f"{cc**2:.2f}"
      pe = f"{calculated_metrics[source]['pe'][idx]:.2f}"
      metrics = f"r$^2$ = ${cc2}$ | pe = ${pe}$"
      if label =='B_H': label = '$\\Delta B_H$'

      if style == 'timeseries':
        if source == 'GMU':
          component_labels2[source][idx] = f"Ref"
        elif source == 'OpenGGCM':
          component_labels2[source][idx] = f"GGCM"
        else:
          component_labels2[source][idx] = f"{source}"

      if style == 'scatter':
        if len(calculated_sources) == 1:
          ylabels.append(f"Calculated {label} {source} [{unit}]")
        else:
          ylabels.append(f"Calculated {label} [{unit}]")
        if source == 'GMU':
          component_labels2[source][idx] = f"Ref    {metrics}"
        elif source == 'OpenGGCM':
          component_labels2[source][idx] = f"GGCM {metrics}"
        else:
          component_labels2[source][idx] = f"{source} {metrics}"
        if cc**2 >= 0.6:
          fit[source] = True

      if cc < 0 and label == 'GIC':
        calculated[source]['data'] = -calculated[source]['data']

  if style == 'timeseries': fit = None

  kwargs = {
    'ylabels': ylabels,
    'component_labels1': component_labels1,
    'component_labels2': component_labels2,
    'suptitle': sid,
    'subplot_label': subplot_label,
    'fit': fit,
    'style': style,
    'show_plots': show_plots
  }

  output_figures = _plot_stack(measured_modified, calculated, **kwargs)
  figures = {}
  for idx, label in enumerate(data['measured'][measured_source]['original']['labels']):
    figures[label] = output_figures[idx]
  return figures


def _plot_measured_original_vs_modified(data, sid, show_plots=False):
  if isinstance(data.get(sid, {}).get('manual_error'), str) or isinstance(data.get(sid, {}).get('automated_error'), str) or 'modified' not in data.keys():
    original = data['original']
    if isinstance(data.get(sid, {}).get('manual_error'), str) or isinstance(data.get(sid, {}).get('automated_error'), str):
      if len(data[sid]['automated_error']) > 0:
        ae = data[sid]['automated_error']
        if isinstance(ae, (list, tuple)):
          data[sid]['automated_error'] = ';\n'.join(map(str, ae))
        else:
          data[sid]['automated_error'] = str(ae)
      suptitle = f"Manual Error: {data[sid]['manual_error']}\nAutomated Error: {data[sid]['automated_error']}"
    else:
      print(original.keys())
      suptitle = f"Modified Error: {original['error']}"
    output_figure = _plot_stack(original, None, ylabels=[f"[{original['unit']}]"], component_labels1=[f"{original['labels'][0]} original"], component_labels2=None,
                suptitle=suptitle, show_plots=show_plots)
    figures = {}
    figures['error'] = output_figure[0]
    return figures
  
  component_labels1 = data['original']['labels'].copy()
  for idx, label in enumerate(component_labels1):
    component_labels1[idx] = f"{label} original"

  component_labels2 = {} #TODO: clean up so don't need modified['modified']
  component_labels2['modified'] = data['modified']['labels'].copy()
  for idx, label in enumerate(component_labels2['modified']):
    component_labels2['modified'][idx] = f"{label} modified"

  kwargs = {
    'ylabels': len(component_labels1)*[f"[{data['original']['unit']}]"],
    'component_labels1': component_labels1,
    'component_labels2': component_labels2,
    'style': 'timeseries',
    'suptitle': sid,
    'show_plots': show_plots
  }

  original = data['original']
  modified = {}
  modified['modified'] = data['modified'] #TODO: clean up so don't need modified['modified']

  kwargs['suptitle'] = f"Modification = {data['modified']['modification']}"
  output_figure = _plot_stack(original, modified, **kwargs)
  figures = {}
  for idx in range(original['data'].shape[1]):
    label = component_labels2['modified'][idx]
    figures[label] = output_figure[idx]
  return figures


def _plot_stack(data1, data2, ylabels, component_labels1, component_labels2, fit=None, suptitle=None, style='timeseries', subplot_label=None, show_plots=False):

  from matplotlib import pyplot as plt
  import pickle
  from datetick import datetick

  from swerve import plt_config, format_cc_scatter, add_subplot_label

  plt.close()
  plt_config()

  line1_opts = {'color': 'black', 'lw': 1}
  line2_opts = {'TVA':{'color': 'blue', 'lw': 0.4},
                'GMU':{'color': 'green', 'lw': 0.4},
                'modified':{'color': 'orange', 'lw': 0.4},
                'SWMF':{'color': 'blue', 'lw': 0.4},
                'MAGE':{'color': 'green', 'lw': 0.4},
                'OpenGGCM':{'color': 'orange', 'lw': 0.4},} #TODO: set outside of function, remove repetition

  n_stack = data1['data'].shape[1]

  if data1 is None:
    return

  if component_labels1 is None:
    component_labels1 = [''] * n_stack
  if component_labels2 is None:
    component_labels2 = [''] * n_stack

  figures = {}
  for j in range(n_stack):
    fig, axes = plt.subplots()

    if j == 0 and suptitle is not None:
      # Use title() for force suptitle to be centered on axis
      axes.set_title(suptitle)

    if ylabels is not None:
      axes.set_ylabel(ylabels[j])

    show_legend = False
    if style == 'timeseries':
      if component_labels1[j]:
        show_legend = True
        kwargs1 = {'label': component_labels1[j], **line1_opts}
      else:
        kwargs1 = line1_opts
      axes.plot(data1['time'], data1['data'][:, j], **kwargs1)

      if data2 is not None:
          for source in data2.keys():
            if component_labels2[source][j]:
              show_legend = True
              kwargs2 = {'label': component_labels2[source][j], **line2_opts[source]}
            else:
              kwargs2 = line2_opts[source]
            axes.plot(data2[source]['time'], data2[source]['data'][:, j], **kwargs2) #TODO: fix issue w time for Bx, By, Bz

      if show_legend:
        leg = axes.legend(ncol=1, frameon=True, loc='upper right')
        for line in leg.get_lines():
          line.set_linewidth(1.5)

    if style == 'scatter':
      for source in data2.keys():
        axes.scatter(data1['data'][:, j], data2[source]['data'][:, j], label=component_labels2[source][j], s=1, color=line2_opts[source]['color'])
        if fit is not None and fit[source] is True:
          import numpy as np
          line_fit = np.polyfit(data1['data'][:, j],  data2[source]['data'][:, j], 1)
          m, b = line_fit
          x_fit = np.array([data1['data'][:, j].min(), data1['data'][:, j].max()])
          y_fit = m * x_fit + b
          axes.plot(x_fit, y_fit, color=line2_opts[source]['color'], linestyle='--', linewidth=1,
             label=f"{source} best-fit slope = {m:.2f}")
      axes.set_xlabel(component_labels1[j])
      format_cc_scatter(axes, regression=True) # setting regression = False centers point (0,0) on plot, saves space
      # TODO: switch for format_cc_scatter
      leg = axes.legend(loc='lower right', handletextpad=0.1, markerscale=3)
      for line in leg.get_lines():
        line.set_linewidth(1.5)

    axes.grid(True)

    if style == 'timeseries':
      datetick('x')

    if subplot_label != None:
      add_subplot_label(axes, subplot_label)

    figures[j] = pickle.dumps(fig)

  if show_plots:
    plt.show()
  plt.close('all')

  return figures


def _save_plots(plots, fname, dir_compare, logger=None, include_label=True):
  import pickle
  from swerve import savefig
  for label, fig in plots.items():
    if include_label:
      fname = f"{label}{fname}"
    fig = pickle.loads(fig)
    savefig(dir_compare, fname, logger=logger, logger_indent=4)
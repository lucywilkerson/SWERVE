# Partial rewrite of linear_regression.py, which has too much code duplication
# and will be very difficult to maintain and generalize.

import os
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from swerve import config, savefig, savefig_paper, add_subplot_label

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
results_dir = os.path.join(CONFIG['dirs']['data'], '_results')
paper_dir = os.path.join(CONFIG['dirs']['paper'], 'figures')
paper_results_dir = os.path.join(paper_dir, '_results')

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

cc_hypothesis_test = True # If true, runs hypothesis test on cc of regression models and returns p-values

plot_types = None # If none, create both line plots and scatter cc plots
if plot_types is None:
  plot_types = ['line', 'scatter']

def df_prep():
  from swerve import read_info_df
  # Read GIC stats or create file if no GIC stats file
  def gic_stats(data_types=['GIC']):
    import pickle
    temp_pkl = os.path.join(results_dir, 'gic_stats.pkl')

    if os.path.exists(temp_pkl):
      # read gic_std and gic_max from it.
      logger.info(f"  Reading gic_std and gic_maxabs from {temp_pkl}")
      with open(temp_pkl, 'rb') as f:
        gic_site, gic_std, gic_maxabs = pickle.load(f)

    else:
      from swerve import site_read, site_stats

      info_df = read_info_df(data_type='GIC', data_class='measured', exclude_errors=True)
      #info_df = read_info()
      sites = info_df['site_id']

      gic_site = []
      gic_std = []
      gic_maxabs = []

      data = {}
      stats = {}
      for sid in sites:

        data[sid] = site_read(sid, data_types=data_types, logger=logger)

        # Extract gic_std and gic_maxabs from existing stats information in data[sid]
        # (no need to add gic_max to stats b/c it can be derived from min/max)
        stats[sid] = site_stats(sid, data[sid], data_types=data_types, logger=logger)

        stat_keys = stats[sid].keys()
        for data_type in stat_keys:
          if data_type != 'GIC/measured/NERC' and data_type != 'GIC/measured/TVA':
            continue
          if not stats[sid][data_type]:
            logger.warning(f"  No stats for {sid}/{data_type}. Skipping.")
            continue
          gic_site.append(sid)
          gic_std.append(stats[sid][data_type]['stats']['std'][0])
          gic_maxabs.append(max(stats[sid][data_type]['stats']['max'], abs(stats[sid][data_type]['stats']['min'])))
      # Save gic_std and gic_maxabs in temp_pkl
      # Ensure the directory exists before saving
      os.makedirs(os.path.dirname(temp_pkl), exist_ok=True)
      with open(temp_pkl, 'wb') as f:
        logger.info(f"  Saving gic_std and gic_maxabs to {temp_pkl}")
        pickle.dump((gic_site, gic_std, gic_maxabs), f)
    return gic_site, gic_std, gic_maxabs 
  
  # Read info and add GIC stats
  info = read_info_df(extended=True, data_type='GIC', data_class='measured', exclude_errors=True)
  sites, gic_std, gic_maxabs = gic_stats()
  info = info[info['site_id'].isin(sites)].reset_index(drop=True)
  info['gic_std'] = gic_std
  info['gic_max'] = gic_maxabs 
  return info

def input_combos(input_set):
  combos = []
  for r in range(1, len(input_set) + 1):
      for combo in itertools.combinations(input_set, r):
        combos.append(list(combo))
  return list(combos)

def regress(x, y):

  def regress_metrics(target, predictions, n_inputs):

    import numpy as np
    def bootstrap_cc_unc(target, predictions, n_bootstrap=1000, random_state=None):
        """
        Returns: 2 sigma uncertainty in cc
        """
        rng = np.random.default_rng(random_state)
        n = len(target)
        cc_samples = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            target_sample = target[idx]
            pred_sample = predictions[idx]
            cc = np.corrcoef(target_sample, pred_sample)[0, 1]
            cc_samples.append(cc)

        cc_samples = np.array(cc_samples)
        #cc_mean = np.mean(cc_samples)
        cc_std = np.std(cc_samples, ddof=1)
        return 2*cc_std

    def cc_95_ci (target, cc):
       #from Devore CH 12.5 (p. 534)
       n = len(target)
       v = np.log((1+cc)/(1-cc))/2 # Fischer transformation
       c1, c2 =(v-(1.96/np.sqrt(n-3)), v+(1.96/np.sqrt(n-3))) # 95% CI endpoints
       ci_lower = (np.exp(2*c1)-1)/(np.exp(2*c1)+1)
       ci_upper = (np.exp(2*c2)-1)/(np.exp(2*c2)+1)
       return ci_lower, ci_upper

    # Calculate error
    rss = np.sum((target-predictions)**2)  # Sum of squares error
    n = len(target)
    rmse = np.sqrt(rss/n)

    # Calculate correlation coefficient
    cc = np.corrcoef(target, predictions)[0,1]
    cc_2se = np.sqrt((1-cc**2)/(n-2)) # see https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
    cc_2se_boot = bootstrap_cc_unc(target, predictions) # bootstrapped uncertainty (2 sigma)
    cc_ci_lower, cc_ci_upper = cc_95_ci(target, cc) # 95% CI for cc from Devore

    # Calculate log likelihood
    n2 = 0.5*n
    llf = -n2*np.log(2*np.pi) - n2*np.log(rss/n) - n2 # see https://stackoverflow.com/a/76135206

    # Calculate AIC and BIC
    k = n_inputs + 1  # number of coefficients + intercept
    aic = -2*llf + 2*k # see https://en.wikipedia.org/wiki/Akaike_information_criterion
    bic = -2*llf + k*np.log(n) # see https://en.wikipedia.org/wiki/Bayesian_information_criterion

    return {'rmse': rmse, 'cc': cc, 'cc_2se': cc_2se, 'cc_ci_lower': cc_ci_lower, 'cc_ci_upper': cc_ci_upper, 'cc_2se_boot': cc_2se_boot, 'aic': aic, 'bic': bic}

  def remove_outliers(data, output, threshold=3.5):
    mask = output <= threshold * np.std(output)
    #num_outliers = len(output) - np.sum(mask)
    output = output[mask]
    x = data[mask]
    return x, output, mask
  
  # Remove outliers
  x, y, mask = remove_outliers(x, y)
  
  model = LinearRegression()
  model.fit(x, y)
  predictions = model.predict(x)
  metrics = regress_metrics(y, predictions, x.shape[1])
  return model, mask, metrics

def plot_line_scatter(x, y, inputs, output_name, mask, model=None, eqn=None, metrics=None):
    #import matplotlib.pyplot as plt
    from swerve import plt_config

    plt_config()
    for i, input_name in enumerate(inputs):
      plt.figure()
      plt.scatter(x[:, i][mask], y[mask], color='k')
      plt.scatter(x[:, i][~mask], y[~mask], facecolors='none', edgecolors='k')
      if model is not None:
        x_range = np.linspace(x[:,i].min(),x[:,i].max(), 100)
        y_model = model.predict(np.column_stack([x_range if j == i else np.zeros_like(x_range) for j in range(x.shape[1])]))
        plt.plot(x_range, y_model, color='m', linewidth=2, linestyle='--', label=f'${eqn}$')
      if metrics is not None:
        text = (
          f"cc = ${metrics['cc']:.2f}$ ± ${metrics['cc_2se_boot']:.2f}$  |  "
          f"RMSE = ${metrics['rmse']:.1f}$ [A]"
        )
        plt.scatter([], [], facecolors='none', edgecolors='none', label=text) # Adds metrics to legend while keeping legend marker alined w eqn
      plt.xlabel(f'${labels.get(input_name, input_name)}$')
      plt.ylabel(f'${labels.get(output_name, output_name)}$ [A]')
      plt.grid(True)
      plt.legend(bbox_to_anchor=(0.005, 0.93), loc='upper left')
      plt.tight_layout()

def plot_cc_scatter(y, predicted, output_name, mask, metrics, eqn):
    from swerve import plt_config, format_cc_scatter
    plt_config()
    plt.figure()
    plt.scatter(y[mask], predicted[mask], color='k')
    plt.scatter(y[~mask], predicted[~mask], facecolors='none', edgecolors='k')
    output_label = (labels.get(output_name, output_name))
    plt.xlabel(f'Measured {output_label} [A]')
    plt.ylabel(f'Predicted {output_label} [A]')
    plt.grid(True)
    #format_cc_scatter(plt.gca()) #TODO: incorporate this into plot_cc_scatter, right now axes are too long
    limits = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    ticks = plt.xticks()[0]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--')
    #^all above lines can be removed once format_cc_scatter is incorporated

    plt.tight_layout()

    text = (
        f"{eqn}\n"
        f"cc = ${metrics['cc']:.2f}$ ± ${metrics['cc_2se_boot']:.2f}$\n"
        f"RMSE = ${metrics['rmse']:.1f}$ [A]\n"
        f"AIC = ${metrics['aic']:.1f}$\n"
        f"BIC = ${metrics['bic']:.1f}$"
    )
    text_kwargs = {
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'fontsize': plt.rcParams['xtick.labelsize'],
        'bbox': {
            "boxstyle": "round,pad=0.3",
            "edgecolor": "black",
            "facecolor": "white",
            "linewidth": 0.5,
            "alpha": 0.7
        }
    }
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, **text_kwargs)

def write_eqn_and_fname(inputs, output_name, model):
    """
    Create a string for the equation and a filename based on inputs, output_name, and model.
    """
    eqn = f"{labels.get(output_name, output_name)} = "
    eqn_txt = f"{output_name} = "
    for i, input_name in enumerate(inputs):
      eqn_txt += f"{model.coef_[i]:+.3g} {input_name}"
      eqn += f"{model.coef_[i]:+.3g} {labels.get(input_name, input_name)} "
    eqn += f" {model.intercept_:+.3g}"
    eqn_txt += f" {model.intercept_:+.3g}"

    fname = f'_fit_{inputs[0]}_{output_name}'
    if '*' in inputs[0]:
        # Create product term and add it to info df
        input1, input2 = inputs[0].split('*')
        fname = f'_fit_{input1}_{input2}_{output_name}'
    return eqn, eqn_txt, fname

def run_cc_hypothesis_test(scatter_fit_df, y, compare_inputs):
  import re
  import scipy.stats as stats

  cc_values = []
  for compare_input in compare_inputs:
    input_str = ', '.join(compare_input)
    row = scatter_fit_df[scatter_fit_df['inputs'] == input_str]
    if not row.empty:
      cc_se_str = row.iloc[0]['cc ± 2SE']
      # Extract cc from string like "$0.85$ ± $0.03$"
      match = re.match(r"\$(.*?)\$ ± \$(.*?)\$", cc_se_str)
      if match:
          cc = float(match.group(1))
          cc_values.append(cc)

  logger.info("Running Fisher-z hyp. test on equality of ccs for models:")
  logger.info(f"  cc = {cc_values[0]:.2f}; inputs = {compare_inputs[0]} ")
  logger.info("  vs")
  logger.info(f"  cc = {cc_values[1]:.2f}; inputs = {compare_inputs[1]} ")

  # https://biostatistics.letgen.org/tag/fishers-z-transformation/
  # Run hypothesis test! Matches http://vassarstats.net/rdiff.html
  V = 0.5*np.log((1+cc_values[0])/(1-cc_values[0]))
  z = (V - 0.5*np.log((1+cc_values[1])/(1-cc_values[1]))) * np.sqrt((len(y)-3)/2)
  logger.info(f"  z = {z:.4f}")
  logger.info(f'  p = {2*stats.norm.sf(abs(z)):.4f} (two-tailed)')
  # Performing z-test for alpha=0.05
  alpha_z = 0.05
  critical_value = 1.96  # Two-tailed test for alpha=0.05
  if abs(z) > critical_value:
      logger.info(f"  Reject null with alpha = {alpha_z})")
  else:
      logger.info(f"  Do not reject null with alpha = {alpha_z}")

def df_sort(scatter_fit_df):
  # Remove any duplicate rows
  scatter_fit_df = scatter_fit_df.drop_duplicates(subset=['inputs'])
  # Remove rows where the input combination is ['alpha', 'alpha*interpolated_beta'] or ['interpolated_beta', 'alpha*interpolated_beta']
  scatter_fit_df = scatter_fit_df[~scatter_fit_df['inputs'].isin(['alpha, alpha*interpolated_beta', 'interpolated_beta, alpha*interpolated_beta'])]
  # Remove rows where the input combination is ['mag_lat', 'mag_lat*interpolated_beta'] or ['interpolated_beta', 'mag_lat*interpolated_beta'], or ['mag_lat*interpolated_beta']
  scatter_fit_df = scatter_fit_df[~scatter_fit_df['inputs'].isin(['mag_lat, mag_lat*interpolated_beta', 'interpolated_beta, mag_lat*interpolated_beta', 'mag_lat*interpolated_beta'])]
  # Move row with inputs = 'mag_lat' (lambda) to the second row
  lambda_row = scatter_fit_df[scatter_fit_df['inputs'] == 'mag_lat']
  scatter_fit_df = scatter_fit_df[scatter_fit_df['inputs'] != 'mag_lat']
  scatter_fit_df = pd.concat([scatter_fit_df.iloc[:1], lambda_row, scatter_fit_df.iloc[1:]], ignore_index=True)
  # Move ['alpha, interpolated_beta'] and ['alpha, interpolated_beta, alpha*interpolated_beta'] rows to the end
  alpha_rows = scatter_fit_df[scatter_fit_df['inputs'].isin(['alpha, interpolated_beta', 'alpha, interpolated_beta, alpha*interpolated_beta'])]
  scatter_fit_df = scatter_fit_df[~scatter_fit_df['inputs'].isin(['alpha, interpolated_beta', 'alpha, interpolated_beta, alpha*interpolated_beta'])]
  scatter_fit_df = pd.concat([scatter_fit_df, alpha_rows], ignore_index=True)
  return scatter_fit_df

# Use the commented out line after info.py has been modified to include gic_max and gic_std.
#output_names = ['gic_max', 'gic_std']
#output_names = ['mag_lat', 'geo_lat']
output_names = ['gic_max']

labels = {
    'mag_lat': '\\lambda',
    'mag_lon': 'Magnetic Longitude [deg]',
    'geo_lat': 'Geographic Latitude [deg]',
    'interpolated_beta': '\\beta',
    'log_beta': '\\log_{10} (\\beta)',
    'alpha': '\\alpha',
    'gic_std': '\\sigma_\\text{GIC} [A]',
    'gic_max': '\\vert{\\text{GIC}\\vert_\\text{max}}',
    'mag_lat*mag_lon': 'Mag. Lat. \\cdot Mag. Long.',
    'alpha*interpolated_beta': '\\alpha \\cdot \\beta',
    'mag_lat*interpolated_beta': '\\lambda \\cdot \\beta',
}

# Each string must appear as a key in labels.
input_sets = [
    #['mag_lat', 'interpolated_beta'],
    #['mag_lat', 'mag_lon'],
    #['mag_lat', 'mag_lon', 'mag_lat*mag_lon'],
    #['alpha', 'interpolated_beta'],
    ['alpha', 'interpolated_beta', 'alpha*interpolated_beta'],
    #['alpha', 'mag_lat'],
    ['mag_lat', 'interpolated_beta', 'mag_lat*interpolated_beta'],

]

paper_inputs = {'alpha':['a)', 'b)'],
                'interpolated_beta':['b)', 'd)'],
                'alpha*interpolated_beta':['c)', 'f)']}

info = df_prep()

for output_name in output_names:
  # Table to hold metrics
  scatter_fit_df = pd.DataFrame(columns=['Fit Equation', 'cc ± 2SE', 'RMSE [A]', 'AIC', 'BIC', 'inputs'])

  for input_set in input_sets:
    for inputs in input_combos(input_set):
      logger.info(f"output = {output_name}")
      logger.info(f"inputs = {inputs}")

      for input_name in inputs:
        if '*' in input_name:
          # Create product term and add it to info df
          input1, input2 = input_name.split('*')
          info[input_name] = info[input1] * info[input2]

      x = info[inputs].values
      y = info[output_name].values

      # Remove rows where y is nan
      mask = ~pd.isna(y)
      x = x[mask]
      y = y[mask]

      # Run regression
      model, mask, metrics = regress(x, y)

      eqn, eqn_txt, base_fname = write_eqn_and_fname(inputs, output_name, model)
      logger.info(f"  Equation: {eqn_txt}")
      for key in metrics:
        logger.info(f"  {key} = {metrics[key]:.4f}")

      # Add metrics to table
      scatter_fit_df.loc[len(scatter_fit_df)] = {
            'Fit Equation': f"${eqn}$",
            'cc ± 2SE': f"${metrics['cc']:.2f}$ ± ${metrics['cc_2se_boot']:.2f}$",
            'RMSE [A]': f"${metrics['rmse']:.1f}$",
            'AIC': f"${metrics['aic']:.1f}$",
            'BIC': f"${metrics['bic']:.1f}$",
            'inputs': ', '.join(inputs)
        }

      # Creating plots
      for plot_type in plot_types:
        if plot_type == 'line': # Plot the scatter plot with regression line
          if len(inputs) == 1:
            plot_line_scatter(x, y, inputs, output_name, mask, model=model, eqn=eqn, metrics=metrics)
            paper_fig_index = 0
          else:
            logger.warning("  Skipping line plot because more than one input.")
        elif plot_type == 'scatter': # Plot the scatter plot of correlation
            predictions = model.predict(x)
            plot_cc_scatter(y, predictions, output_name, mask, metrics, eqn)
            paper_fig_index = 1

        fname = f'{plot_type}{base_fname}'
        #savefig(results_dir, fname, logger)
        if len(inputs) == 1 and inputs[0] in paper_inputs.keys():
            add_subplot_label(plt.gca(), paper_inputs[inputs[0]][paper_fig_index], loc=(-0.15, 1))
            savefig_paper(paper_results_dir, fname, logger) # TODO: make this cleaner pls
        plt.close()

  # Reorganize output table
  scatter_fit_df = df_sort(scatter_fit_df)

  if cc_hypothesis_test:
    for input_pair in itertools.combinations(scatter_fit_df['inputs'], 2):
      # Run hypothesis test on cc of regression models
      run_cc_hypothesis_test(scatter_fit_df, y, [[input_pair[0]], [input_pair[1]]])
    # Can check with https://www.danielsoper.com/statcalc/calculator.aspx?id=104 
    # Note that Devore yields slightly different results due to no sample size in rho

  # Remove inputs column and adjust indexing
  scatter_fit_df = scatter_fit_df.drop(columns=['inputs'])
  scatter_fit_df.index = scatter_fit_df.index + 1

  # Save output table
  scatter_fit_df.to_markdown(os.path.join(results_dir, f"fit_table_{output_name}.md"), index=True)
  scatter_fit_df.to_latex(os.path.join(results_dir, f"fit_table_{output_name}.tex"), index=True, escape=False)

  # Save output to paper dir
  #scatter_fit_df.to_latex(os.path.join(paper_dir, f"fit_table_{output_name}.tex"), index=True, escape=False)

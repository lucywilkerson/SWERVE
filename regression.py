# Partial rewrite of linear_regression.py, which has too much code duplication
# and will be very difficult to maintain and generalize.

reparse = False  # Reparse the data files, even if they already exist (use if site_read.py modified).
outliers = False  # Remove outliers from regression based on threshold.

import os
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from swerve import cli, config, savefig, savefig_paper, add_subplot_label, fix_latex

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
results_dir = os.path.join(CONFIG['dirs']['data'], '_results', 'regression')

args = cli('regression.py') 
if args['error_type'] == 'manual':
   error_type = 'manual_error'
elif args['error_type'] =='automated':
   error_type = 'automated_error'
else:
    raise ValueError(f"Invalid error-type '{args['error-type']}'. Must be 'manual' or 'automated'.")

if 'paper' in CONFIG['dirs']:
  paper_dir = os.path.join(CONFIG['dirs']['paper'], 'figures')
  paper_results_dir = os.path.join(paper_dir, '_results')
  paper_error_type = 'manual_error'  # Use manual errors for paper figures and tables

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

r_hypothesis_test = True # If true, runs hypothesis test on r of regression models and returns p-values

plot_types = None # If none, create both line plots and scatter r plots
if plot_types is None:
  plot_types = ['line', 'scatter']

def df_prep(reparse=False):
  from swerve import read_info_df
  # Read GIC stats or create file if no GIC stats file
  def gic_stats(data_types=['GIC'], reparse=False):
    import pickle
    temp_pkl = os.path.join(results_dir, f'gic_stats_{error_type}.pkl')

    if os.path.exists(temp_pkl) and not reparse:
      # read gic_std and gic_max from it.
      logger.info(f"  Reading gic_std and gic_maxabs from {temp_pkl}")
      with open(temp_pkl, 'rb') as f:
        gic_site, gic_std, gic_maxabs = pickle.load(f)

    else:
      from swerve import sids, site_read, site_stats

      sites = sids(extended=True, data_type='GIC', data_class='measured', exclude_errors=True, error_type=error_type)

      gic_site = []
      gic_std = []
      gic_maxabs = []

      data = {}
      stats = {}
      for sid in sites:
        sid = str(sid)
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
          gic_maxabs.append(max(stats[sid][data_type]['stats']['max'], abs(stats[sid][data_type]['stats']['min']))[0])
      # Save gic_std and gic_maxabs in temp_pkl
      # Ensure the directory exists before saving
      os.makedirs(os.path.dirname(temp_pkl), exist_ok=True)
      with open(temp_pkl, 'wb') as f:
        logger.info(f"  Saving gic_std and gic_maxabs to {temp_pkl}")
        pickle.dump((gic_site, gic_std, gic_maxabs), f)
    return gic_site, gic_std, gic_maxabs 

  # Read info and add GIC stats
  info = read_info_df(extended=True, data_type='GIC', data_class='measured', exclude_errors=True, error_type=error_type)
  sites, gic_std, gic_maxabs = gic_stats(reparse=reparse)
  info = info[info['site_id'].isin(sites)].reset_index(drop=True)
  info['gic_std'] = gic_std
  info['gic_max'] = gic_maxabs 
  return info

def scatter_fit_df(rows, columns):

  df_raw = pd.DataFrame(rows, columns=columns)

  # Create df for md and latex output
  columns = ['Fit Equation', 'r $\pm$ 2SE', 'r$^2$', 'RMSE [A]', 'AIC', 'inputs']
  # Ideally would determine column indices from df_raw in case columns changes.
  for i, row in enumerate(rows):
    rows[i] = [f"${row[0]}$", f"${row[1]:.2f} \pm {row[2]:.2f}$", f"${row[1]**2:.2f}$", f"${row[3]:.1f}$", f"${row[4]:.1f}$", row[7]]

  df = pd.DataFrame(rows, columns=columns)
  # Remove any duplicate rows
  df = df.drop_duplicates(subset=['inputs'])

  # Remove rows with these input combinations
  # TODO: simplify this by setting outside of function, also figure out duplicates situation
  omits = ['mag_lat',
           'mag_lat, interpolated_beta',
           'mag_lat, mag_lat*interpolated_beta',
           'mag_lat, mag_lat*interpolated_beta',
           'interpolated_beta, mag_lat*interpolated_beta',
           'mag_lat, interpolated_beta, mag_lat*interpolated_beta',
           'mag_lat*interpolated_beta'
          ]
  df = df[~df['inputs'].isin(omits)]

  # Move row with inputs = 'mag_lat' (lambda) to the second row
  lambda_row = df[df['inputs'] == 'mag_lat']
  df = df[df['inputs'] != 'mag_lat']
  df = pd.concat([df.iloc[:1], lambda_row, df.iloc[1:]], ignore_index=True)

  # Move rows with any alpha inputs to the end
  alpha_inputs = ['alpha, interpolated_beta', 'alpha, interpolated_beta, alpha*interpolated_beta']
  alpha_rows = df[df['inputs'].isin(alpha_inputs)]
  df = df[~df['inputs'].isin(alpha_inputs)]
  df = pd.concat([df, alpha_rows], ignore_index=True)

  # Adjust indexing
  df = df.drop(columns=['inputs'])
  df.index = df.index + 1

  return df_raw, df

def input_combos(input_set):
  combos = []
  for r in range(1, len(input_set) + 1):
      for combo in itertools.combinations(input_set, r):
        combos.append(list(combo))
  return list(combos)

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
          f"r = ${metrics['r']:.2f}$ ± ${metrics['r_2se']:.2f}$  |  "
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
    plt.xlabel(f'Measured ${output_label}$ [A]')
    plt.ylabel(f'Predicted ${output_label}$ [A]')
    plt.grid()
    format_cc_scatter(plt.gca(), regression=True)

    plt.tight_layout()

    text = (
        f"${eqn}$\n"
        f"r = ${metrics['r']:.2f}$ ± ${metrics['r_2se']:.2f}$\n"
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

  r_values = []
  aic_values = []
  for compare_input in compare_inputs:
    input_str = ', '.join(compare_input)
    row = scatter_fit_df[scatter_fit_df['inputs'] == input_str]
    aic_values.append(row.iloc[0]['AIC'])
    r_values.append(row.iloc[0]['r'])

  logger.info("Running Fisher-z hyp. test on equality of rs for models:")
  logger.info(f"  r = {r_values[0]:.2f}; AIC = {aic_values[0]}; inputs = {compare_inputs[0]} ")
  logger.info("  vs")
  logger.info(f"  r = {r_values[1]:.2f}; AIC = {aic_values[1]}; inputs = {compare_inputs[1]} ")

  # Note that Devore yields slightly different results due to no sample size in rho
  # https://biostatistics.letgen.org/tag/fishers-z-transformation/
  # Run hypothesis test! Matches http://vassarstats.net/rdiff.html
  # Can also check with https://www.danielsoper.com/statcalc/calculator.aspx?id=104 
  V = 0.5*np.log((1+r_values[0])/(1-r_values[0]))
  z = (V - 0.5*np.log((1+r_values[1])/(1-r_values[1]))) * np.sqrt((len(y)-3)/2)
  logger.info(f"  z = {z:.4f}")
  logger.info(f'  p = {2*stats.norm.sf(abs(z)):.4f} (two-tailed)')

  # Performing z-test for alpha=0.05
  alpha_z = 0.05
  critical_value = 1.96  # Two-tailed test for alpha=0.05
  if abs(z) > critical_value:
      logger.info(f"  Reject null with alpha = {alpha_z})")
  else:
      logger.info(f"  Do not reject null with alpha = {alpha_z}")

def regress(x, y):

  def regress_metrics(target, predictions, n_inputs):

    import numpy as np
    from scipy import stats
    def bootstrap_r_unc(target, predictions, n_bootstrap=1000, random_state=None):
        """
        Returns: 2 sigma uncertainty in r
        """
        rng = np.random.default_rng(random_state)
        n = len(target)
        r_samples = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            target_sample = target[idx]
            pred_sample = predictions[idx]
            r = np.corrcoef(target_sample, pred_sample)[0, 1]
            r_samples.append(r)

        r_samples = np.array(r_samples)
        #r_mean = np.mean(r_samples)
        r_std = np.std(r_samples, ddof=1)
        return 2*r_std

    def r_95_ci (target, r):
       #from Devore CH 12.5 (p. 534)
       n = len(target)
       v = np.log((1+r)/(1-r))/2 # Fischer transformation
       c1, c2 =(v-(1.96/np.sqrt(n-3)), v+(1.96/np.sqrt(n-3))) # 95% CI endpoints
       ci_lower = (np.exp(2*c1)-1)/(np.exp(2*c1)+1)
       ci_upper = (np.exp(2*c2)-1)/(np.exp(2*c2)+1)
       return ci_lower, ci_upper
    
    def calc_p_values(x, target, model):
      x_with_intercept = np.column_stack([np.ones(x.shape[0]), x])
      params = np.append(model.intercept_, model.coef_)
      y_hat = np.dot(x_with_intercept, params)
      residuals = target - y_hat
      dof = x_with_intercept.shape[0] - x_with_intercept.shape[1]
      mse = np.sum(residuals**2) / dof
      var_b = mse * np.linalg.inv(np.dot(x_with_intercept.T, x_with_intercept)).diagonal()
      se_b = np.sqrt(var_b)
      t_stats = params / se_b
      p_values = [2 * (1 - stats.t.cdf(np.abs(t), dof)) for t in t_stats]
      return p_values[1:]  # exclude intercept


    # Calculate error
    rss = np.sum((target-predictions)**2)  # Sum of squares error
    n = len(target)
    rmse = np.sqrt(rss/n)

    # Calculate correlation coefficient
    r = np.corrcoef(target, predictions)[0,1]
    r_2se = np.sqrt((1-r**2)/(n-2)) # see https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
    r_2se_boot = bootstrap_r_unc(target, predictions) # bootstrapped uncertainty (2 sigma)
    r_ci_lower, r_ci_upper = r_95_ci(target, r) # 95% CI for r from Devore

    # Calculate log likelihood
    n2 = 0.5*n
    llf = -n2*np.log(2*np.pi) - n2*np.log(rss/n) - n2 # see https://stackoverflow.com/a/76135206

    # Calculate AIC and BIC
    k = n_inputs + 1  # number of coefficients + intercept
    aic = -2*llf + 2*k # see https://en.wikipedia.org/wiki/Akaike_information_criterion
    bic = -2*llf + k*np.log(n) # see https://en.wikipedia.org/wiki/Bayesian_information_criterion

    # Calculate p-value for each input coefficient
    p_values = calc_p_values(x, target, model)

    return {'rmse': rmse, 'r': r, 'r_2se': r_2se, 'r_ci_lower': r_ci_lower, 'r_ci_upper': r_ci_upper, 'r_2se_boot': r_2se_boot, 'aic': aic, 'bic': bic, 'p_values': p_values}

  def remove_outliers(data, output, threshold=3.5):
    mask = output <= threshold * np.std(output)
    #num_outliers = len(output) - np.sum(mask)
    output = output[mask]
    x = data[mask]
    return x, output, mask
  
  # Remove outliers
  if outliers:
    x, y, mask = remove_outliers(x, y)
  
  model = LinearRegression()
  model.fit(x, y)
  predictions = model.predict(x)
  metrics = regress_metrics(y, predictions, x.shape[1])
  return model, mask, metrics


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

if 'paper' in CONFIG['dirs'] and error_type == paper_error_type:
  paper_inputs = {'alpha':['a)', 'b)'],
                  'interpolated_beta':['b)', 'd)'],
                  'alpha*interpolated_beta':['c)', 'f)']}
else:
  paper_inputs = {}

info = df_prep(reparse=reparse)

columns = ['Fit Equation', 'r', '2SE', 'RMSE [A]', 'AIC', 'BIC', 'p-values', 'inputs']
for output_name in output_names:
  # Table to hold metrics
  scatter_fit_df_rows = []

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
        if key != 'p_values':
          logger.info(f"  {key} = {metrics[key]:.4f}")
        else:
          for i, p_value in enumerate(metrics['p_values']):
            logger.info(f"  p-value for {inputs[i]} = {p_value:.4e}")

      # Modify if columns above changes
      row = [eqn, metrics['r'], metrics['r_2se'], metrics['rmse'], metrics['aic'], metrics['bic'], metrics['p_values'], ', '.join(inputs)]
      scatter_fit_df_rows.append(row)

      # Creating plots
      for plot_type in plot_types:
        if plot_type == 'line': # Plot the scatter plot with regression line
          if len(inputs) == 1:
            plot_line_scatter(x, y, inputs, output_name, mask, model=model, eqn=eqn, metrics=metrics)
            paper_fig_index = 0
          else:
            logger.warning("  Skipping line plot because more than one input.")
            continue
        elif plot_type == 'scatter': # Plot the scatter plot of correlation
            predictions = model.predict(x)
            plot_cc_scatter(y, predictions, output_name, mask, metrics, eqn)
            paper_fig_index = 1

        fname = f'{plot_type}{base_fname}'
        savefig(results_dir, f'{fname}_{error_type}', logger)
        if len(inputs) == 1 and inputs[0] in paper_inputs.keys():
            add_subplot_label(plt.gca(), paper_inputs[inputs[0]][paper_fig_index], loc=(-0.15, 1))
            savefig_paper(paper_results_dir, fname, logger) 
        plt.close()

  # Convert to df and create df_table for markdown and latex output
  df, df_table = scatter_fit_df(scatter_fit_df_rows, columns)

  if r_hypothesis_test:
    for input_pair in itertools.combinations(df['inputs'], 2):
      # Run hypothesis test on r of regression models
      run_cc_hypothesis_test(df, y, [[input_pair[0]], [input_pair[1]]])


  # Save output table
  fname = os.path.join(results_dir, f"fit_table_{output_name}_{error_type}")
  logger.info(f"Writing {fname}_{error_type}.md")
  df_table.to_markdown(fname + ".md", index=True)
  latex_str = fix_latex(df_table, data_type='fit', index=True)
  with open(fname + ".tex", "w") as f:
        logger.info(f"Writing {fname}_{error_type}.tex")
        f.write(latex_str)

  # Save output to paper dir
  if 'paper' in CONFIG['dirs'] and error_type == paper_error_type:
    fname = os.path.join(paper_dir, f"fit_table_{output_name}.tex")
    with open(fname, "w") as f:
          f.write(latex_str)
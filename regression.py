# Partial rewrite of linear_regression.py, which has too much code duplication
# and will be very difficult to maintain and generalize.

# First modify info.py to put log10_beta, alpha, std, and gic_max in info.extended.csv

import os
import itertools
import pandas as pd

from sklearn.linear_model import LinearRegression

from swerve import LOG_KWARGS, logger

logger = logger(**LOG_KWARGS)

def read_info():
  file_path = os.path.join('info', 'info.extended.csv')
  info = pd.read_csv(file_path)
  # Filter out sites with error message
  # Also remove rows that don't have data_type = GIC and data_class = measured
  info = info[~info['error'].str.contains('', na=False)]
  info = info[info['data_type'].str.contains('GIC', na=False)]
  info = info[info['data_class'].str.contains('measured', na=False)]
  info.reset_index(drop=True, inplace=True)
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

    # Calculate error
    rss = np.sum((target-predictions)**2)  # Sum of squares error
    n = len(target)
    rms = np.sqrt(rss/n)

    # Calculate correlation coefficient
    cc = np.corrcoef(target, predictions)[0,1]
    cc_2se = np.sqrt((1-cc**2)/(n-2)) # see https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
    cc_2se_boot = bootstrap_cc_unc(target, predictions) # bootstrapped uncertainty (2 sigma)

    # Calculate log likelihood
    n2 = 0.5*n
    llf = -n2*np.log(2*np.pi) - n2*np.log(rss/n) - n2 # see https://stackoverflow.com/a/76135206

    # Calculate AIC and BIC
    k = n_inputs + 1  # number of coefficients + intercept
    aic = -2*llf + 2*k # see https://en.wikipedia.org/wiki/Akaike_information_criterion
    bic = -2*llf + k*np.log(n) # see https://en.wikipedia.org/wiki/Bayesian_information_criterion

    return {'rms': rms, 'cc': cc, 'cc_2se': cc_2se, 'cc_2se_boot': cc_2se_boot, 'aic': aic, 'bic': bic}

  model = LinearRegression()
  model.fit(x, y)
  predictions = model.predict(x)
  metrics = regress_metrics(y, predictions, x.shape[1])
  return model, metrics

def plot_scatter(x, y, inputs, output_name):
    import matplotlib.pyplot as plt

    for i, input_name in enumerate(inputs):
      plt.figure(figsize=(8, 6))
      plt.scatter(x[:, i], y, alpha=0.7)
      plt.xlabel(labels.get(input_name, input_name))
      plt.ylabel(labels.get(output_name, output_name))
      plt.grid(True)
      plt.tight_layout()
      plt.show()

# Use the commented out line after info.py has been modified to include gic_max and std.
#output_names = ['gic_max', 'std']
output_names = ['mag_lat', 'geo_lat']

labels = {
    'mag_lat': 'Magnetic Latitude [deg]',
    'mag_lon': 'Magnetic Longitude [deg]',
    'geo_lat': 'Geographic Latitude [deg]',
    'interpolated_beta': r'$\beta$',
    'log_beta': r'$\log_{10} (\beta)$',
    'alpha': r'$\alpha$',
    'std': r'$\sigma_\text{GIC}$ [A]',
    'gic_max': r'$\vert{\text{GIC}_\text{max}}\vert$',
    'mag_lat*mag_lon': r'Mag. Lat. $\cdot$ Mag. Long.',
}

# Each string must appear as a key in labels.
input_sets = [
    ['mag_lat', 'interpolated_beta'],
    ['mag_lat', 'mag_lon'],
    ['mag_lat', 'mag_lon', 'mag_lat*mag_lon'],
]

info = read_info()

for output_name in output_names:
  for input_set in input_sets:
    for inputs in input_combos(input_set):
      logger.info(f"output = {output_name}; inputs = {inputs}")

      for input_name in inputs:
        if '*' in input_name:
          # Create product term and add it to info df
          input1, input2 = input_name.split('*')
          info[input_name] = info[input1] * info[input2]

      x = info[inputs].values
      y = info[output_name].values

      #plot_scatter(x, y, inputs, output_name)
      model, metrics = regress(x, y)

      eqn = f"{output_name} = "
      for i, input_name in enumerate(inputs):
        eqn += f"{model.coef_[i]:+.3g}*{input_name} "
      eqn += f" {model.intercept_:+.3g}"
      logger.info(f"  Equation: {eqn}")
      for key in metrics:
        logger.info(f"  {key} = {metrics[key]:.4f}")

utilrsw.rm_if_empty('log/regression.errors.log')
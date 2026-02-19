import numpy as np
from sklearn.linear_model import LinearRegression

def regress(x, y, outliers=False):

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
  else:
    mask = np.array([True]*len(y))
  
  model = LinearRegression()
  model.fit(x, y)
  predictions = model.predict(x)
  metrics = regress_metrics(y, predictions, x.shape[1])
  return model, mask, metrics

def write_eqn_and_fname(inputs, output_name, model, labels):
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
        if 'slope' in inputs[0]:
          # Create product term with slope and add it to info df
          input1, input2, input3 = inputs[0].split('*')
          fname = f'_fit_{input2}_{input3}_{output_name}'
        else:
          # Create product term and add it to info df
          input1, input2 = inputs[0].split('*')
          fname = f'_fit_{input1}_{input2}_{output_name}'
    return eqn, eqn_txt, fname
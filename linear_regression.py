import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from sklearn.metrics import mean_squared_error
import pickle

from swerve import FILES, plt_config, savefig, savefig_paper, subset, add_subplot_label

import os
import matplotlib.pyplot as plt
import datetime
import json
from itertools import combinations

limits = plt_config()

results_dir = os.path.join('..', '2024-May-Storm-data', '_results')
cc_path = os.path.join(results_dir, 'cc.pkl')
all_dir  = os.path.join('..', '2024-May-Storm-data', '_all')
all_file = os.path.join(all_dir, 'all.pkl')

# Configure logging
import utilrsw
logger = utilrsw.logger(log_dir=os.path.join('log'))

def load_data(file_path):
    """Load data from a pickle or CSV file."""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return pd.DataFrame(data)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pkl or .csv file.")

def read(all_file, sid=None):
  fname = os.path.join('info', 'info.json')
  with open(fname, 'r') as f:
    logger.info(f"Reading {fname}")
    info_dict = json.load(f)

  info_df = pd.read_csv(os.path.join('info', 'info.csv'))

  fname = os.path.join('info', 'plot.json')
  with open(fname, 'r') as f:
    logger.info(f"Reading {fname}")
    plot_cfg = json.load(f)

  logger.info(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data, plot_cfg

fmts = ['png','pdf']
def savefig(fdir, fname, fmts=fmts):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        logger.info(f"    Saving {fname}.{fmt}")
        plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join('..','2024-May-Storm-paper', sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    logger.info(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

cc_compare = False #perform regression of cc
std_compare = True #perform regression of std
peak_compare = True #perform regression of peak GIC
log10_beta = False #use log10 of beta instead of beta
alpha = False #use alpha instead of lat
z_test = True #perform z-test on beta vs log10(beta)

paper=True
if paper:
    cc_compare = False # cc compare is not included in paper analysis

def subset(time, data, start, stop):
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

start = datetime.datetime(2024, 5, 10, 15, 0)
stop = datetime.datetime(2024, 5, 12, 6, 0)

def find_target(target_name):
    if target_name == 'cc':
        target_label = '|cc|'
        target_symbol = r'|cc|'
    elif target_name =='std':
        target_label = 'Standard Deviation [A]'
        target_symbol = r'$\sigma$'
    elif target_name == 'gic_max':
        target_label = 'Peak GIC [A]'
        target_symbol = r'$\vert{\text{GIC}_\text{max}}\vert$'
    else:
        target_label = target_name
        target_symbol = 'y'
    return target_label, target_symbol
    

def analyze_fit(target, predictions, features):
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
    cc_unc = np.sqrt((1-cc**2)/(n-2)) # see https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
    cc_unc = bootstrap_cc_unc(target, predictions) # bootstrapped uncertainty (2 sigma)

    # Calculate log likelihood
    n2 = 0.5*n
    llf = -n2*np.log(2*np.pi) - n2*np.log(rss/n) - n2 # see https://stackoverflow.com/a/76135206

    # Calculate AIC and BIC
    k = len(features) + 1  # number of coefficients + intercept
    aic = -2*llf + 2*k # see https://en.wikipedia.org/wiki/Akaike_information_criterion
    bic = -2*llf + k*np.log(n) # see https://en.wikipedia.org/wiki/Bayesian_information_criterion

    return rss, rms, cc, cc_unc, aic, bic

def remove_outliers(data, target, features, threshold=3.5):
    mask = target <= threshold * np.std(target)
    #num_outliers = len(target) - np.sum(mask)
    data = data[mask]
    target = target[mask]
    if features == 'log_beta':
        x = np.log10(data['interpolated_beta'])
    elif features == 'alpha':
        x = .001*np.exp(.115*data['mag_lat']) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
    else:
        x = data[features]
    return x, target, mask

def plot_regression(target, predictions, remove_outlier, target_label, mask):

    plt.figure()
    if remove_outlier:
        outlier_mask = mask
        plt.scatter(target[outlier_mask], predictions[outlier_mask], color='k', alpha=0.9, label='Predicted vs Actual')
        plt.scatter(target[~outlier_mask], predictions[~outlier_mask], facecolors='none', edgecolors='k', alpha=0.9, label='Omitted Points')
    else:
        plt.scatter(target, predictions, color='k', alpha=0.9, label='Predicted vs Actual')

    plt.plot([target.min(), target.max()], [target.min(), target.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
    plt.xlabel(f'Measured {target_label}')
    plt.ylabel(f'Predicted {target_label}')
    limits = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    ticks = plt.xticks()[0]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.grid()

def plot_residuals(target, predictions,  remove_outlier, target_label, mask, fit_eqn, n_bins=25):
    # Plotting histogram of residuals (errors)
    plt.figure()
    if remove_outlier:
        residuals_masked = predictions[mask] - target[mask]
        residuals_outliers = predictions[~mask] - target[~mask]
        # finding bin edges
        all_residuals = np.concatenate([residuals_masked, residuals_outliers])
        bins = np.histogram_bin_edges(all_residuals, bins=n_bins)
        # setting bins to be the same
        plt.hist(residuals_masked, bins=bins, color='gray', edgecolor='k', alpha=0.7)
        if len(residuals_outliers) > 0:
            plt.hist(residuals_outliers, bins=bins, color='white', edgecolor='k', label='Outlier')
    else:
        residuals = predictions - target
        plt.hist(residuals, bins=20, color='gray', edgecolor='k', alpha=0.7)
    plt.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel(f'Residual (Predicted - Measured) {target_label}')
    plt.ylabel('Count')
    plt.grid()
    plt.legend(title=fit_eqn, title_fontsize=plt.rcParams['legend.fontsize'])

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

def add_text(target_symbol, features, slope, intercept, rms, cc, cc_unc, aic, bic, df=None):
    def get_feature_symbol(feature):
        feature_str = str(feature)
        if feature_str == 'mag_lat':
            return r'$\lambda$'
        elif feature_str == 'geo_lat':
            return r'$\lambda_{geo}$'
        elif feature_str == 'interpolated_beta':
            return r'$\beta$'
        elif feature_str == 'log_beta':
            return r'$\log_{10} (\beta)$'
        elif feature_str == 'alpha':
            return r'$\alpha$'
        elif '*' in feature_str:
            parts = feature_str.split('*')
            symbols = [get_feature_symbol(part) for part in parts]
            return ''.join(symbols)
        else:
            return feature_str
    
    def write_eqn(target_symbol, features, slope, intercept):
        feature_symbols = []
        for feature in features:
            symbol = get_feature_symbol(feature)
            feature_symbols.append(symbol)
        # Formatting  fit eqn string for arbitrary number of features and cross terms
        if isinstance(slope, (float, int, np.floating, np.integer)):
            # Single feature
            fit_eqn = f"{target_symbol} = ${slope:.2f}$ {', '.join(feature_symbols)} ${intercept:+.2f}$"
        elif hasattr(slope, '__iter__'):
            # Multiple features
            terms = []
            for coef, symbol in zip(slope, feature_symbols):
                terms.append(f"${coef:+.2f}$ {symbol}")
            fit_eqn = f"{target_symbol} = " + " ".join(terms)
            if isinstance(intercept, (float, int, np.floating, np.integer)):
                fit_eqn += f" ${intercept:+.2f}$"
        else:
            fit_eqn = f"{target_symbol} = ..."
        return fit_eqn

    fit_eqn = write_eqn(target_symbol, features, slope, intercept)

    if df is not None:
        df.loc[len(df)] = {
            'Fit Equation': fit_eqn,
            'cc ± 2SE': f"${cc:.2f}$ ± ${cc_unc:.2f}$",
            'RMS [A]': f"${rms:.2f}$",
            'AIC': f"${aic:.1f}$",
            'BIC': f"${bic:.1f}$"
        }

    text = (
        f"{fit_eqn}\n"
        f"cc = ${cc:.2f}$ ± ${cc_unc:.2f}$\n"
        f"RMS = ${rms:.1f}$ [A]\n"
        f"AIC = ${aic:.1f}$\n"
        f"BIC = ${bic:.1f}$"
    )
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, **text_kwargs)
    return fit_eqn

def linear_regression_model(data, features, feature_names, target, target_name, remove_outlier=True, df=None, plot_fit=False, plot_rms=False):
    
    def plot_1D_fit(data, feature, target, predictions, remove_outlier, mask, target_label, feature_label, fit_eqn):
        plt.figure()
        if feature == 'log_beta':
                x = np.log10(data['interpolated_beta'])
        if feature == 'alpha':
            x = .001*np.exp(.115*data['mag_lat']) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
        else:
            x = data[[feature]]
        if remove_outlier and mask is not None:
            plt.scatter(x[mask], target[mask], color='k', label='Data')
            plt.scatter(x[~mask], target[~mask], facecolors='none', edgecolors='k', label='Outlier')
        else:
            plt.scatter(x, target, color='k', label='Target')
        # Sort for line plot
        sort_idx = np.argsort(x)
        x1d = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
        sort_idx = np.argsort(x1d)
        if fit_eqn is not None:
            plt.plot(x1d.iloc[sort_idx] if hasattr(x1d, 'iloc') else x1d[sort_idx], predictions[sort_idx], color='m', linestyle='--', label=fit_eqn)
        else:
            plt.plot(x1d.iloc[sort_idx] if hasattr(x1d, 'iloc') else x1d[sort_idx], predictions[sort_idx], color='m', linestyle='--', label='Prediction')
        plt.xlabel(feature_label if feature_label else feature)
        plt.ylabel(target_label if target_label else "Target")
        plt.legend(loc='upper left')
        plt.grid()

    models = {}
    errors = {}

    for feature in features:
        if remove_outlier:
            x, y, mask = remove_outliers(data, target, feature)
        else:
            if feature == 'log_beta':
                x = np.log10(data['interpolated_beta'])
            if feature == 'alpha':
                x = .001*np.exp(.115*data['mag_lat']) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
            else:
                x = data[[feature]]
            y = target
            mask = None
        
        model = LinearRegression()
        model.fit(pd.DataFrame(x).values.reshape(-1, 1), y)
        
        predictions = model.predict(x.values.reshape(-1, 1))
        
        rss, rms, cc, cc_unc, aic, bic = analyze_fit(y, predictions, feature)
        #cc = np.corrcoef(y, predictions)[0,1]
        
        logger.info(f"Linear Regression for {target_name} with only {feature}:")
        logger.info(f"  Coefficient: {model.coef_[0]}")
        logger.info(f"  Intercept: {model.intercept_}")
        logger.info(f"  RSS: {rss}")
        logger.info(f"  RMS: {rms}")
        logger.info("\n")
        
        # Store model and error
        models[feature] = model
        errors[feature] = rms

        target_label, target_symbol = find_target(target_name)
        
        # Plot actual vs predicted values
        if feature == 'log_beta':
            predictions = model.predict(np.log10(data['interpolated_beta']).values.reshape(-1, 1))
        elif feature == 'alpha':
            a = .001*np.exp(.115*data['mag_lat']) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
            predictions = model.predict(a.values.reshape(-1, 1))
        else:
            predictions = model.predict(data[[feature]].values.reshape(-1, 1))
    
        plot_regression(target, predictions, remove_outlier, target_label, mask=mask)
        if target_name == 'cc':
            plt.title(f"Linear Regression for {feature_names.get(feature, feature)}\nRMS Error: {rms:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
            plt.text(0.05, 0.95, cc, transform=plt.gca().transAxes, **text_kwargs)
            fit_eqn = None
        else:
            fit_eqn = add_text(target_symbol, [feature], model.coef_[0], model.intercept_, rms, cc, cc_unc, aic, bic, df=df)
        savefig(results_dir, 'scatter_fit_' + feature + '_' + target_name)
        if paper:
            text = None
            if target_name == 'std':
                text = 'a)' if feature == 'mag_lat' else 'c)' if feature == 'interpolated_beta' else None
            elif target_name == 'gic_max':
                text = 'b)' if feature == 'mag_lat' else 'd)' if feature == 'interpolated_beta' else None
            if text is not None:
                add_subplot_label(plt.gca(), text)
                savefig_paper('scatter_fit_' + feature + '_' + target_name, sub_dir="regression_model")
        plt.close()

        if plot_fit:
            plot_1D_fit(data, feature, target, predictions, remove_outlier, mask, target_label, feature_names.get(feature, feature), fit_eqn)
            savefig(results_dir, 'line_fit_' + feature + '_' + target_name)
            plt.close()
        if plot_rms:
            plot_residuals(target, predictions, remove_outlier, target_label, mask, fit_eqn)
            savefig(results_dir, 'residual_hist_' + feature + '_' + target_name)
            plt.close()

    
    return models, errors

def linear_regression_all(data, features, target, target_name, remove_outlier=True, df=None, plot_rms=False):
    """Perform linear regression using all features."""
    
    if remove_outlier:
        x, y, mask = remove_outliers(data, target, features)
    else:
        x = data[features]
        y = target
        mask = None
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Make predictions
    predictions = model.predict(x)
    
    rss, rms, cc, cc_unc, aic, bic = analyze_fit(y, predictions, features)
    
    # Print model coefficients and error
    logger.info(f"Linear Regression with All Features for {target}:")
    for feature, coef in zip(features, model.coef_):
        logger.info(f"  Coefficient for {feature}: {coef}")
    logger.info("  Intercept:", model.intercept_)
    logger.info("  RSS:", rss)
    logger.info("  RMS:", rms)
    logger.info("\n")

    target_label, target_symbol = find_target(target_name)

    # Plot actual vs predicted values
    predictions = model.predict(data[features])
    plot_regression(target, predictions, remove_outlier, target_label, mask=mask)
    if target_name == 'cc':
        plt.title(f"Linear Regression for {feature_names.get(feature, feature)}\nRMS Error: {rms:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
        plt.text(0.05, 0.95, cc, transform=plt.gca().transAxes, **text_kwargs)
        fit_eqn = None
    else:
        fit_eqn = add_text(target_symbol, features, model.coef_, model.intercept_, rms, cc, cc_unc, aic, bic, df=df)
    savefig(results_dir, f'scatter_fit_all_{target_name}')
    if paper:
        text = None
        if target_name == 'std':
                text = 'e)'
        elif target_name == 'gic_max':
                text = 'f)'
        add_subplot_label(plt.gca(), text)
        savefig_paper(f'scatter_fit_all_{target_name}', sub_dir="regression_model")
    plt.close()

    if plot_rms:
        plot_residuals(target, predictions, remove_outlier, target_label, mask, fit_eqn)
        savefig(results_dir, 'residual_hist_all_' + target_name)
        plt.close()
    
    return model, rms

def linear_regression_cross(data, features, target, target_name, remove_outlier=True, df=None, plot_rms=False):
    """Perform linear regression with all combinations of features and calculate AIC and BIC for each model."""

    results = []
    # Generate all possible combinations of features, including cross terms
    for r in range(1, len(features) + 1):
        for combo in combinations(features, r):
            # Prepare the data for the current combination of features
            combo_features = list(combo)
            cross_terms = [f"{f1}*{f2}" for i, f1 in enumerate(combo_features) for f2 in combo_features[i+1:]]
            for cross_term in cross_terms:
                f1, f2 = cross_term.split('*')
                data.loc[:, cross_term] = data[f1] * data[f2]
            all_features = combo_features + cross_terms

            if remove_outlier:
                x, y, mask = remove_outliers(data, target, all_features)
            else:
                x = data[all_features]
                y = target
                mask = None

            # Fit the linear regression model
            model = LinearRegression()
            model.fit(x, y)

            # Make predictions
            predictions = model.predict(x)

            rss, rms, cc, cc_unc, aic, bic = analyze_fit(y, predictions, all_features)

            result_entry = {
                'model': model,
                'features': all_features,
                'rss': rss,
                'rms': rms,
                'aic': aic,
                'bic': bic,
                'cc': cc,
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }

            if remove_outlier:
                result_entry['mask'] = mask
            results.append(result_entry)

            # Clean up cross terms from the data
            for cross_term in cross_terms:
                del data[cross_term]

    # Sort results by AIC (ascending)
    results = sorted(results, key=lambda x: x['aic'])

    # Extract the best model based on AIC
    best_model = results[0]
    model = best_model['model']
    rss = best_model['rss']
    rms = best_model['rms']
    aic = best_model['aic'] 
    bic = best_model['bic']
    cc = best_model['cc']
    all_features = best_model['features']
    coefficients = best_model['coefficients']
    intercept = best_model['intercept']

    # Print results for the best model
    logger.info(f"Best Linear Regression Model with Cross Terms (Based on AIC) for {target}:")
    logger.info("  Coefficients:")
    for feature, coef in zip(all_features, coefficients):
        logger.info(f"    Coefficient for {feature}: {coef}")
    logger.info("  Intercept:", intercept)
    logger.info("  RSS:", rss)
    logger.info("  RMS:", rms)
    logger.info("  AIC:", aic)
    logger.info("  BIC:", bic)
    logger.info("  Correlation Coefficient (cc):", cc)
    logger.info("\n")

    target_label, target_symbol = find_target(target_name)
    
    # Add cross-term features to the DataFrame
    for cross_term in all_features:
        if '*' in cross_term:
            f1, f2 = cross_term.split('*')
            data[cross_term] = data[f1] * data[f2]
    predictions = model.predict(data[all_features])
    # Clean up cross-term features from the DataFrame
    for cross_term in all_features:
        if '*' in cross_term:
            del data[cross_term]

    # Plot actual vs predicted values for the best model
    plot_regression(target, predictions, remove_outlier, target_label, mask=mask)
    if target_name == 'cc':
        plt.title(f"Best Linear Regression with {', '.join(all_features)}\nRMS Error: {rms:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
        plt.text(0.05, 0.95, cc, transform=plt.gca().transAxes, **text_kwargs)
        fit_eqn = None
    else:
        fit_eqn = add_text(target_symbol, all_features, coefficients, intercept, rms, cc, cc_unc, aic, bic, df=df)
    savefig(results_dir, f'scatter_fit_cross_{target_name}')
    if paper:
        text = None
        if target_name == 'std':
                text = 'e)'
        elif target_name == 'gic_max':
                text = 'f)'
        add_subplot_label(plt.gca(), text)
        savefig_paper(f'scatter_fit_cross_{target_name}', sub_dir="regression_model")
    plt.close()

    if plot_rms:
        plot_residuals(target, predictions, remove_outlier, target_label, mask, fit_eqn)
        savefig(results_dir, 'residual_hist_cross_' + target_name)
        plt.close()
    
    return results


if cc_compare:
    # Load the data
    data = load_data(cc_path)
    
    features = ['dist(km)', 'log_beta_diff', 'lat_diff']
    feature_names = {
            'dist(km)': 'Distance [km]',
            'log_beta_diff': r'|$\Delta \log_{10} (\beta)$|',
            'lat_diff': r'$\Delta$ Latitude [deg]'
        }
    
    target = np.abs(data[['cc']])
    target_name = 'cc'

    """#test cc calc
    x=target
    y=data[['dist(km)']]
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    cc = np.corrcoef(y, predictions)[0,1]
    exit()"""
    
    # Perform linear regression
    # TODO: fix errors in cc linear regression
    #model, error = linear_regression_model(data, features, feature_names=feature_names, target=target, target_name=target_name, remove_outlier=False)
    all_features_model, all_features_error = linear_regression_all(data, features=features, target=target, target_name=target_name, remove_outlier=False)
    models_aic_bic = linear_regression_cross(data, features=features, target=target, target_name=target_name, remove_outlier=False)



if std_compare or peak_compare:
    # Load the data
    file_path = os.path.join('info', 'info.extended.csv')
    data = load_data(file_path)
    # Filter out sites with error message
    # Also remove rows that don't have data_type = GIC and data_class = measured
    data = data[~data['error'].str.contains('', na=False)]
    data = data[data['data_type'].str.contains('GIC', na=False)]
    data = data[data['data_class'].str.contains('measured', na=False)]
    data.reset_index(drop=True, inplace=True)
    sites = data['site_id'].tolist()

    info_dict, info_df, data_all, plot_info = read(all_file)

    if not log10_beta and not alpha:
        features = ['mag_lat', 'interpolated_beta']
        feature_names = {
                'mag_lat': 'Latitude [deg]',
                'interpolated_beta': r'$\beta$'
            }
        
        for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
            if compare:
                target = np.zeros(len(sites))
                for i, sid in enumerate(sites):
                    time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                    data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                    time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                    target[i] = func(data_meas[~np.isnan(data_meas)])

                scatter_fit_df = pd.DataFrame(columns=['Fit Equation', 'cc ± 2SE', 'RMS [A]', 'AIC', 'BIC'])

                model, error = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name, df=scatter_fit_df, plot_fit=True)
                all_features_model, all_features_error = linear_regression_all(data, features=features, target=target, target_name=target_name, df=scatter_fit_df)
                models_aic_bic = linear_regression_cross(data, features=features, target=target, target_name=target_name)

                #print(scatter_fit_df)
                scatter_fit_df.to_markdown(os.path.join(results_dir, f"fit_table_{target_name}.md"), index=False)
                scatter_fit_df.to_latex(os.path.join(results_dir, f"fit_table_{target_name}.tex"), index=False, escape=False)
    elif not alpha:
        features = ['mag_lat', 'log_beta']
        feature_names = {
                'mag_lat': 'Latitude [deg]',
                'log_beta': r'$\log_{10} (\beta)$'
            }
        
        for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
            if compare:
                target = np.zeros(len(sites))
                for i, sid in enumerate(sites):
                    time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                    data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                    time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                    target[i] = func(data_meas[~np.isnan(data_meas)])

                model, error = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name, plot_fit=True)
    elif not log10_beta:
        features = ['alpha', 'interpolated_beta']
        feature_names = {
                'alpha': r'$\alpha$',
                'interpolated_beta': r'$\beta$'
            }
        
        for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
            if compare:
                target = np.zeros(len(sites))
                for i, sid in enumerate(sites):
                    time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                    data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                    time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                    target[i] = func(data_meas[~np.isnan(data_meas)])

                model, error = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name, plot_fit=True)
    
if z_test:
    if not std_compare and not peak_compare:
        # Load the data
        file_dir = os.path.join('..', '2024-May-Storm', 'info')
        file_path = os.path.join(file_dir, 'info.extended.csv')
        data = load_data(file_path)
        # Filter out sites with error message
        # Also remove rows that don't have data_type = GIC and data_class = measured
        data = data[~data['error'].str.contains('', na=False)]
        data = data[data['data_type'].str.contains('GIC', na=False)]
        data = data[data['data_class'].str.contains('measured', na=False)]
        data.reset_index(drop=True, inplace=True)
        sites = data['site_id'].tolist()

        info_dict, info_df, data_all, plot_info = read(all_file)

    features = ['interpolated_beta', 'log_beta']
    feature_names = {
                'interpolated_beta': r'$\beta$',
                'log_beta': r'$\log_{10} (\beta)$'
            }
    
    for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
                target = np.zeros(len(sites))
                for i, sid in enumerate(sites):
                    time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                    data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                    time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                    target[i] = func(data_meas[~np.isnan(data_meas)])

                models, errors = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name, plot_fit=False, plot_rms=False)
                # Save mean and std of predicted values to an array
                predictions_beta = models[features[0]].predict(data[[features[0]]].values.reshape(-1, 1))
                predictions_log_beta = models[features[1]].predict(np.log10(data[[features[0]]]).values.reshape(-1, 1))

                pred_stats = np.array([
                    [np.mean(predictions_beta), np.std(predictions_beta, ddof=1)],
                    [np.mean(predictions_log_beta), np.std(predictions_log_beta, ddof=1)]
                ])

                # Calculate z-scores
                z_score = (pred_stats[0,0] - pred_stats[0,1]) / np.sqrt((pred_stats[1,0]**2 + pred_stats[1,1]**2) / np.sqrt(len(data[[features[0]]])))
    
                logger.info(f"Z-score for beta vs log10(beta): {z_score}")

                # Performing z-test for alpha=0.01
                alpha_z = 0.01
                critical_value = 2.576  # Two-tailed test for alpha=0.01
                if abs(z_score) > critical_value:
                    logger.info(f"Reject null hypothesis: significant difference between beta and log10(beta) at alpha={alpha_z}")
                else:
                    logger.info(f"Fail to reject null hypothesis: no significant difference between beta and log10(beta) at alpha={alpha_z}")
    
    features = ['mag_lat', 'alpha']
    feature_names = {
                'mag_lat': r'$\lambda$',
                'alpha': r'$\alpha$'
            }
    
    for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
                target = np.zeros(len(sites))
                for i, sid in enumerate(sites):
                    time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                    data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                    time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                    target[i] = func(data_meas[~np.isnan(data_meas)])

                models, errors = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name, plot_fit=False, plot_rms=False)
                # Save mean and std of predicted values to an array
                predictions_lat = models[features[0]].predict(data[[features[0]]].values.reshape(-1, 1))
                a = .001*np.exp(.115*data[[features[0]]]) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
                predictions_alpha = models[features[1]].predict(a.values.reshape(-1, 1))

                pred_stats = np.array([
                    [np.mean(predictions_lat), np.std(predictions_lat, ddof=1)],
                    [np.mean(predictions_alpha), np.std(predictions_alpha, ddof=1)]
                ])

                # Calculate z-scores
                z_score = (pred_stats[0,0] - pred_stats[0,1]) / np.sqrt((pred_stats[1,0]**2 + pred_stats[1,1]**2) / np.sqrt(len(data[[features[0]]])))
    
                logger.info(f"Z-score for lambda vs alpha: {z_score}")

                # Performing z-test for alpha=0.01
                alpha_z = 0.01
                critical_value = 2.576  # Two-tailed test for alpha=0.01
                if abs(z_score) > critical_value:
                    logger.info(f"Reject null hypothesis: significant difference between lambda and alpha at alpha={alpha_z}")
                else:
                    logger.info(f"Fail to reject null hypothesis: no significant difference between lambda and alpha at alpha={alpha_z}")

utilrsw.rm_if_empty('log/linear_regression.errors.log')
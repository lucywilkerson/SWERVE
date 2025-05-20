import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

import os
import matplotlib.pyplot as plt
import datetime
import json
from itertools import combinations
import logging

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600


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

results_dir = os.path.join('..', '2024-May-Storm-data', '_results')

# Configure logging
log_file = os.path.join(results_dir, 'linear_regression.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

fmts = ['png','pdf']
def savefig(fdir, fname, fmts=fmts):
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

cc_compare = True #perform regression of cc
std_compare = True #perform regression of std
peak_compare = True #perform regression of peak GIC

paper=True
if paper:
    cc_compare = False # cc compare is not included in paper analysis

    def add_subplot_label(ax, label, loc=(-0.15, 1)):
        ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

def subset(time, data, start, stop):
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

start = datetime.datetime(2024, 5, 10, 15, 0)
stop = datetime.datetime(2024, 5, 12, 6, 0)

def find_label(target_name):
    if target_name == 'cc':
        target_label = '|cc|'
    elif target_name =='std':
        target_label = 'Standard Deviation [A]'
    elif target_name == 'gic_max':
        target_label = 'Peak GIC [A]'
    else:
        target_label = target_name
    return target_label
    

def analyze_fit(target, predictions, features):
    # Calculate error
    rss = np.sum((target - predictions) ** 2)  # Sum of squares error
    n = len(target)
    rms = np.sqrt(rss/n)

    # Calculate correlation coefficient
    cc = np.corrcoef(target, predictions)[0,1]

    # Calculate AIC and BIC
    k = len(features) + 1  # Number of coefficients + intercept
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)

    return rss, rms, cc, aic, bic

def remove_outliers(data, target, features, threshold=3.5):
    mask = target <= threshold * np.std(target)
    #num_outliers = len(target) - np.sum(mask)
    data = data[mask]
    target = target[mask]
    return data[features], target, mask

def plot_regression(target, predictions, cc, remove_outlier, target_label, mask):

    def add_cc_text(cc):
        text_kwargs = {
            'horizontalalignment': 'left',
            'verticalalignment': 'top',
            'fontsize': plt.rcParams['xtick.labelsize'],
            'bbox': {
            "boxstyle": "round,pad=0.3",
            "edgecolor": "black",
            "facecolor": "white",
            "linewidth": 0.5
            }
        }
        text = f"cc = {cc:.2f}"
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, **text_kwargs)

    plt.figure()
    if remove_outlier:
        outlier_mask = mask
        plt.scatter(target[outlier_mask], predictions[outlier_mask], color='k', alpha=0.9, label='Predicted vs Actual')
        plt.scatter(target[~outlier_mask], predictions[~outlier_mask], facecolors='none', edgecolors='k', alpha=0.9, label='Omitted Points')
    else:
        plt.scatter(target, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
    add_cc_text(cc)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
    plt.xlabel(f'Measured {target_label}')
    plt.ylabel(f'Predicted {target_label}')
    plt.grid()

def linear_regression_model(data, features, feature_names, target, target_name, remove_outlier=True, plot=True):
    
    models = {}
    errors = {}

    for feature in features:
        if remove_outlier:
            x, y, mask = remove_outliers(data, target, feature)
        else:
            x = data[[feature]]
            y = target
            mask = None
        
        model = LinearRegression()
        model.fit(pd.DataFrame(x).values.reshape(-1, 1), y)
        
        predictions = model.predict(x.values.reshape(-1, 1))

        print(np.corrcoef(y, predictions)[0,1])
        if np.corrcoef(y, predictions)[0,1] == np.nan:
            exit()
        
        rss, rms, cc, aic, bic = analyze_fit(y, predictions, feature)
        #cc = np.corrcoef(y, predictions)[0,1]
        
        logging.info(f"Linear Regression for {target_name} with only {feature}:")
        logging.info(f"  Coefficient: {model.coef_[0]}")
        logging.info(f"  Intercept: {model.intercept_}")
        logging.info(f"  RSS: {rss}")
        logging.info(f"  RMS: {rms}")
        logging.info("\n")
        
        # Store model and error
        models[feature] = model
        errors[feature] = rms

        target_label = find_label(target_name)
        
        # Plot actual vs predicted values
        predictions = model.predict(data[[feature]])
        if plot:
            plot_regression(target, predictions, cc, remove_outlier, target_label, mask=mask)
        plt.title(f"Linear Regression for {feature_names.get(feature, feature)}\nRMS Error: {rms:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
        savefig(results_dir, 'scatter_fit_' + feature + '_' + target_name)
        if paper:
            text = None
            if target_name == 'std':
                text = 'a)' if feature == 'geo_lat' else 'c)' if feature == 'interpolated_beta' else None
            elif target_name == 'gic_max':
                text = 'b)' if feature == 'geo_lat' else 'd)' if feature == 'interpolated_beta' else None
            add_subplot_label(plt.gca(), text)
            savefig_paper('scatter_fit_' + feature + '_' + target_name, sub_dir="regression_model")
        plt.close()
    
    return models, errors

def linear_regression_all(data, features, target, target_name, remove_outlier=True):
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
    
    rss, rms, cc, aic, bic = analyze_fit(y, predictions, features)
    
    # Print model coefficients and error
    logging.info(f"Linear Regression with All Features for {target}:")
    for feature, coef in zip(features, model.coef_):
        logging.info(f"  Coefficient for {feature}: {coef}")
    logging.info("  Intercept:", model.intercept_)
    logging.info("  RSS:", rss)
    logging.info("  RMS:", rms)
    logging.info("\n")

    target_label = find_label(target_name)

    # Plot actual vs predicted values
    predictions = model.predict(data[features])
    plot_regression(target, predictions, cc, remove_outlier, target_label, mask=mask)
    plt.title(f"Linear Regression for {', '.join(features)}\nRMS Error: {float(rms):.2f}\nAIC: {float(aic):.2f}, BIC: {float(bic):.2f}")
    #plt.legend()
    savefig(results_dir, f'scatter_fit_all_{target_name}')
    plt.close()
    
    return model, rms

def linear_regression_cross(data, features, target, target_name, remove_outlier=True):
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

            rss, rms, cc, aic, bic = analyze_fit(y, predictions, all_features)

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
    logging.info(f"Best Linear Regression Model with Cross Terms (Based on AIC) for {target}:")
    logging.info("  Coefficients:")
    for feature, coef in zip(all_features, coefficients):
        logging.info(f"    Coefficient for {feature}: {coef}")
    logging.info("  Intercept:", intercept)
    logging.info("  RSS:", rss)
    logging.info("  RMS:", rms)
    logging.info("  AIC:", aic)
    logging.info("  BIC:", bic)
    logging.info("  Correlation Coefficient (cc):", cc)
    logging.info("\n")

    target_label = find_label(target_name)
    
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
    plot_regression(target, predictions, cc, remove_outlier, target_label, mask=mask)
    plt.title(f"Best Linear Regression with {', '.join(all_features)}\nRMS Error: {rms:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
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
    
    return results


if cc_compare:
    # Load the data
    file_dir = os.path.join('..', '2024-May-Storm-data', '_results')
    file_path = os.path.join(file_dir, 'cc.pkl')
    data = load_data(file_path)
    
    features = ['dist(km)', 'beta_diff', 'lat_diff']
    feature_names = {
            'dist(km)': 'Distance [km]',
            'beta_diff': r'|$\Delta \log_{10} (\beta)$|',
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

    all_dir  = os.path.join('..', '2024-May-Storm-data', '_all')
    all_file = os.path.join(all_dir, 'all.pkl')
    info_dict, info_df, data_all, plot_info = read(all_file)

    features = ['geo_lat', 'interpolated_beta']
    feature_names = {
            'geo_lat': 'Latitude [deg]',
            'interpolated_beta': r'|$\log_{10} (\beta)$|'
        }
    
    for compare, target_name, func in [(std_compare, 'std', np.std), (peak_compare, 'gic_max', lambda x: max(np.abs(x)))]:
        if compare:
            target = np.zeros(len(sites))
            for i, sid in enumerate(sites):
                time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                target[i] = func(data_meas[~np.isnan(data_meas)])
            model, error = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name)
            if not paper:
                all_features_model, all_features_error = linear_regression_all(data, features=features, target=target, target_name=target_name)
            models_aic_bic = linear_regression_cross(data, features=features, target=target, target_name=target_name)

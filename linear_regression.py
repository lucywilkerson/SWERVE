import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

import os
import matplotlib.pyplot as plt
import datetime
import json

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
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

fmts = ['png','pdf']
def savefig(fdir, fname, fmts=fmts):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

cc_compare = True #perform regression of cc
std_compare = True #perform regression of std
peak_compare = True #perform regression of peak GIC

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

def add_cc_text(cc):
    text_kwargs = {
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'bbox': {
            "boxstyle": "round,pad=0.3",
            "edgecolor": "black",
            "facecolor": "white",
            "linewidth": 0.5
        }
    }
    text = f"cc = {cc:.2f}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, **text_kwargs)

def linear_regression_model(data, features, feature_names, target, target_name):
    
    models = {}
    errors = {}

    for feature in features:
        x = data[[feature]]
        y = target
        
        model = LinearRegression()
        model.fit(x, y)
        
        predictions = model.predict(x)
        error = np.sum((y - predictions) ** 2)  # Sum of squares error

        # Calculate correlation coefficient
        cc = np.corrcoef(y, predictions)[0,1]
        
        print(f"Linear Regression for {feature}:")
        print("  Coefficient:", model.coef_[0])
        print("  Intercept:", model.intercept_)
        print("  Sum of Squares Error:", error)
        print()
        
        # Store model and error
        models[feature] = model
        errors[feature] = error

        target_label = find_label(target_name)
        
        # Plot actual vs predicted values
        plt.figure()
        plt.scatter(y, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
        add_cc_text(cc)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
        plt.xlabel(f'Actual {target_label}')
        plt.ylabel(f'Predicted {target_label}')
        plt.title(f"Linear Regression for {feature_names.get(feature, feature)}\nSum of Squares Error: {error:.2f}")
        plt.grid()
        savefig(results_dir, 'scatter_fit_' + feature + '_' + target_name)
        plt.close()
    
    return models, errors

def linear_regression_all(data, features, target, target_name):
    """Perform linear regression using all features."""
    
    # Prepare the data
    x = data[features]
    y = target
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Make predictions
    predictions = model.predict(x)
    
    # Calculate least squares error
    error = np.sum((y - predictions) ** 2)  # Sum of squares error

    # Calculate correlation coefficient
    cc = np.corrcoef(y, predictions)[0,1]
    
    # Print model coefficients and error
    print("Linear Regression with All Features:")
    for feature, coef in zip(features, model.coef_):
        print(f"  Coefficient for {feature}: {coef}")
    print("  Intercept:", model.intercept_)
    print("  Sum of Squares Error:", error)
    print()

    target_label = find_label(target_name)

    # Plot actual vs predicted values
    plt.figure()
    plt.scatter(y, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
    add_cc_text(cc)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
    plt.xlabel(f'Actual {target_label}')
    plt.ylabel(f'Predicted {target_label}')
    plt.title(f"Linear Regression for All Features\nSum of Squares Error: {error:.2f}")
    plt.grid()
    #plt.legend()
    savefig(results_dir, f'scatter_fit_all_{target_name}')
    plt.close()
    
    return model, error

def linear_regression_cross(data, features, target, target_name):
    """Perform linear regression with cross terms and calculate AIC and BIC."""
    results = {}
    
    # Generate cross terms
    cross_terms = []
    for i in range(len(features)):
        for j in range(i, len(features)):
            cross_term = f"{features[i]}*{features[j]}"
            data[cross_term] = data[features[i]] * data[features[j]]
            cross_terms.append(cross_term)
    
    all_features = features + cross_terms

    # Prepare the data
    x = data[all_features]
    y = target
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Make predictions
    predictions = model.predict(x)
    
    # Calculate residual sum of squares
    rss = np.sum((y - predictions) ** 2)
    
    # Number of observations and parameters
    n = len(y)
    k = len(all_features) + 1  # Number of coefficients + intercept
    
    # Calculate AIC and BIC
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)

    # Calculate correlation coefficient
    cc = np.corrcoef(y, predictions)[0,1]
    
    # Store results
    results['model'] = model
    results['rss'] = rss
    results['aic'] = aic
    results['bic'] = bic
    results['coefficients'] = model.coef_
    results['intercept'] = model.intercept_
    
    # Print results
    print("Linear Regression with Cross Terms:")
    for feature, coef in zip(all_features, model.coef_):
        print(f"  Coefficient for {feature}: {coef}")
    print("  Intercept:", model.intercept_)
    print("  RSS:", rss)
    print("  AIC:", aic)
    print("  BIC:", bic)
    print()

    target_label = find_label(target_name)
    
    # Plot actual vs predicted values
    plt.figure()
    plt.scatter(y, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
    add_cc_text(cc)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
    plt.xlabel(f'Actual {target_label}')
    plt.ylabel(f'Predicted {target_label}')
    plt.title(f"Linear Regression with Cross Terms\nSum of Squares Error: {rss:.2f}\nAIC: {aic:.2f}, BIC: {bic:.2f}")
    plt.grid()
    savefig(results_dir, f'scatter_fit_cross_{target_name}')
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
    
    target = np.abs(data['cc'])
    target_name = 'cc'
    
    # Perform linear regression
    model, error = linear_regression_model(data, features=features, feature_names=feature_names, target=target, target_name=target_name)
    all_features_model, all_features_error = linear_regression_all(data, features=features, target=target, target_name=target_name)
    models_aic_bic = linear_regression_cross(data, features=features, target=target, target_name=target_name)

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
            all_features_model, all_features_error = linear_regression_all(data, features=features, target=target, target_name=target_name)
            models_aic_bic = linear_regression_cross(data, features=features, target=target, target_name=target_name)

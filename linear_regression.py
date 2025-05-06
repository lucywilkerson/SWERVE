import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return pd.DataFrame(data)

results_dir = os.path.join('..', '2024-May-Storm-data', '_results')

fmts = ['png','pdf']
def savefig(fdir, fname, fmts=fmts):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')


features = ['dist(km)', 'beta_diff', 'lat_diff']

feature_names = {
        'dist(km)': 'Distance [km]',
        'beta_diff': r'|$\Delta \log_{10} (\beta)$|',
        'lat_diff': r'$\Delta$ Latitude [deg]'
    }

def linear_regression_model(data, features=features, feature_names=feature_names, target='cc'):
    
    models = {}
    errors = {}

    for feature in features:
        x = data[[feature]]
        y = np.abs(data[target])
        
        model = LinearRegression()
        model.fit(x, y)
        
        predictions = model.predict(x)
        error = np.sum((y - predictions) ** 2)  # Sum of squares error
        
        print(f"Linear Regression for {feature}:")
        print("  Coefficient:", model.coef_[0])
        print("  Intercept:", model.intercept_)
        print("  Sum of Squares Error:", error)
        print()
        
        # Store model and error
        models[feature] = model
        errors[feature] = error
        
        # Plot actual vs predicted values
        plt.figure()
        plt.scatter(y, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual |cc|')
        plt.ylabel('Predicted |cc|')
        plt.title(f"Linear Regression for {feature_names.get(feature, feature)}\nSum of Squares Error: {error:.2f}")
        plt.grid()
        savefig(results_dir, 'scatter_fit_' + feature)
        plt.close()
    
    return models, errors

def linear_regression_all(data, features=features, feature_names=feature_names, target='cc'):
    """Perform linear regression using all features."""
    
    # Prepare the data
    x = data[features]
    y = np.abs(data[target])
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Make predictions
    predictions = model.predict(x)
    
    # Calculate least squares error
    error = np.sum((y - predictions) ** 2)  # Sum of squares error
    
    # Print model coefficients and error
    print("Linear Regression with All Features:")
    for feature, coef in zip(features, model.coef_):
        print(f"  Coefficient for {feature}: {coef}")
    print("  Intercept:", model.intercept_)
    print("  Sum of Squares Error:", error)
    print()
    
    # Plot actual vs predicted values
    plt.figure()
    plt.scatter(y, predictions, color='k', alpha=0.9, label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color=3*[0.6], linewidth=0.5, linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual |cc|')
    plt.ylabel('Predicted |cc|')
    plt.title(f"Linear Regression for All Features\nSum of Squares Error: {error:.2f}")
    plt.grid()
    #plt.legend()
    savefig(results_dir, 'scatter_fit_all')
    plt.close()
    
    return model, error

def main():
    # Load the data
    results_dir = os.path.join('..', '2024-May-Storm-data', '_results')
    file_path = os.path.join(results_dir, 'cc.pkl')
    data = load_data(file_path)
    
    # Perform linear regression
    model, error = linear_regression_model(data)
    all_features_model, all_features_error = linear_regression_all(data)

if __name__ == "__main__":
    main()
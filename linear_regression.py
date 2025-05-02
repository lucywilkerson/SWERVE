import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return pd.DataFrame(data)

def linear_regression_model(data):
    """Perform linear regression and compute error."""
    X = data[['dist(km)', 'beta_diff', 'lat_diff']]
    y = data['cc']
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    error = np.sum((y - predictions) ** 2)  # Sum of squares error
    
    print("Linear Regression Coefficients:", model.coef_)
    print("Linear Regression Intercept:", model.intercept_)
    print("Sum of Squares Error (Linear Regression):", error)
    
    return model, error

def plot_3d_regression(data, model):
    """Plot 3D regression model."""
    X = data[['dist(km)', 'beta_diff', 'lat_diff']]
    y = data['cc']
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of actual data
    ax.scatter(data['dist(km)'], data['beta_diff'], data['lat_diff'], c=y, cmap='viridis', label='Actual Data')
    
    # Create a grid for predictions
    x_range = np.linspace(X['dist(km)'].min(), X['dist(km)'].max(), 10)
    y_range = np.linspace(X['beta_diff'].min(), X['beta_diff'].max(), 10)
    z_range = np.linspace(X['lat_diff'].min(), X['lat_diff'].max(), 10)
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range)
    
    # Flatten the grid and make predictions
    grid_points = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    predictions = model.predict(grid_points)
    
    # Reshape predictions to match the grid
    predictions = predictions.reshape(x_grid.shape)
    
    # Plot the regression surface
    ax.plot_surface(x_grid, y_grid, predictions, color='blue', alpha=0.5, label='Regression Surface')
    
    ax.set_xlabel('dist(km)')
    ax.set_ylabel('beta_diff')
    ax.set_zlabel('lat_diff')
    plt.title('3D Regression Model')
    plt.legend()
    plt.show()

def main():
    # Load the data
    results_dir = os.path.join('..', '2024-May-Storm-data', '_results')
    file_path = os.path.join(results_dir, 'cc.pkl')
    data = load_data(file_path)
    
    # Perform linear regression
    model, error = linear_regression_model(data)
    
    # Plot 3D regression
    plot_3d_regression(data, model)

if __name__ == "__main__":
    main()
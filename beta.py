import os
import csv
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import geopandas as gpd
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator

data_dir = os.path.join('..', '2024-AGU-data')
# Load a MATLAB file
fname = os.path.join(data_dir, 'pulkkinen', 'waveforms_All.mat')
data = loadmat(fname)
data = data['waveform'][0]

# Convert the MATLAB data to a pandas DataFrame
raw_df = pd.DataFrame(data)
#mea_dat = raw_df[0][0][0][0][1][0]
ott_dat = raw_df[0][1][0][0][1][0]
#mmb_dat = raw_df[0][2][0][0][1][0]
#nur_dat = raw_df[0][3][0][0][1][0]

# Convert ott_dat to a pandas DataFrame
rows = []
for i in range(len(ott_dat)):
    beta = ott_dat[i][0][0][1][0][0]
    lat = ott_dat[i][0][0][2][0][0]
    lon = ott_dat[i][0][0][3][0][0]
    rows.append([beta, lat, lon])
df = pd.DataFrame(rows, columns=['beta', 'lat', 'lon'])

# Create the interpolator
interpolator = LinearNDInterpolator(df[['lat', 'lon']], df['beta'])

# Define a grid for interpolation
lat_grid = np.linspace(df['lat'].min(), df['lat'].max(), 100)
lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 100)
lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

# Interpolate the data
beta_grid = interpolator(lat_grid, lon_grid)

# Reading in info.csv
fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Add a new column to info_df based on the interpolated beta values
info_df['interpolated_beta'] = interpolator(info_df['geo_lat'], info_df['geo_lon'])

# Save the updated DataFrame 
out_fname = os.path.join('info', 'info.csv')
info_df.to_csv(out_fname, index=False)
print(f"Saving updated {out_fname}")
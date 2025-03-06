import os
import pickle

import numpy as np
import pandas as pd
import numpy.ma as ma

def read_TVA_or_NERC(row): #can read anything!
    site_id = row['site_id']
    if row['data_type'] == 'gic':
        if row['data_class'] == 'measured':
            if row['data_source'] == 'NERC':
                #reading in data for site if NERC
                fname = os.path.join(data_dir, site_id, 'B_measured_NERC.pkl')
            elif row['data_source'] == 'TVA':
                #reading in data for site if TVA
                site_id = "".join(site_id.split()) #removing space from name to match file name
                fname = os.path.join(data_dir, site_id, 'B_measured_TVA.pkl')

            with open(fname, 'rb') as f:
                #print(f"Reading {fname}")
                site_data = pickle.load(f)

            site_df = pd.DataFrame(site_data)
            data = site_df['modified'][0]['data'] # 1-min avg data
            masked_data = ma.masked_invalid(data) # 1-min data w nan values masked
        elif row['data_class'] == 'calculated':
            if row['data_source'] == 'NERC':
                #reading in data for site if NERC
                fname = os.path.join(data_dir, site_id, 'GIC_calculated_NERC.pkl')
            elif row['data_source'] == 'TVA':
                #reading in data for site if TVA
                site_id = "".join(site_id.split())
                fname = os.path.join(data_dir, site_id, 'GIC_calculated_TVA.pkl')
            
            with open(fname, 'rb') as f:
                #print(f"Reading {fname}")
                site_data = pickle.load(f)
            site_df = pd.DataFrame(site_data)
            data = site_df['original'][0]['data']
            masked_data = ma.masked_invalid(data)
    elif row['data_type'] == 'B':
        if row['data_class'] == 'measured':
            if row['data_source'] == 'NERC':
                #reading in data for site if NERC
                fname = os.path.join(data_dir, site_id, 'B_measured_NERC.pkl')
            elif row['data_source'] == 'TVA':
                #reading in data for site if TVA
                site_id = "".join(site_id.split())
                fname = os.path.join(data_dir, site_id, 'B_measured_TVA.pkl')
            with open(fname, 'rb') as f:
                #print(f"Reading {fname}")
                site_data = pickle.load(f)
            site_df = pd.DataFrame(site_data)
            data = site_df['modified'][0]['data']
            masked_data = ma.masked_invalid(data)
        elif row['data_class'] == 'calculated':
            if row['data_source'] == 'SWMF':
                #reading in data for site if SWMF
                site_id = "".join(site_id.split())
                fname = os.path.join(data_dir, site_id, 'B_calculated_SWMF.pkl')
                with open(fname, 'rb') as f:
                    #print(f"Reading {fname}")
                    site_data = pickle.load(f)
                site_df = pd.DataFrame(site_data)
                data = site_df['original'][0]['data']
                masked_data = ma.masked_invalid(data)
            elif row['data_source'] == 'MAGE':
                #reading in data for site if MAGE
                site_id = "".join(site_id.split())
                fname = os.path.join(data_dir, site_id, 'B_calculated_MAGE.pkl')
                with open(fname, 'rb') as f:
                    #print(f"Reading {fname}")
                    site_data = pickle.load(f)
                site_df = pd.DataFrame(site_data)
                data = site_df['original'][0]['data']
                masked_data = ma.masked_invalid(data)
            
    else:
        print('oh no! the code! its broken!')
        exit()
    return data, masked_data


data_dir = os.path.join('..', '2024-AGU-data', '_processed')
out_dir = os.path.join('..', '2024-AGU-data', '_results')

fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out sites with error message
# Also remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[~info_df['error'].str.contains('', na=False)]
info_df = info_df[info_df['data_type'].str.contains('B', na=False)]
#info_df = info_df[info_df['data_class'].str.contains('calculated', na=False)]
info_df.reset_index(drop=True, inplace=True)
#print(info_df)

sites = info_df['site_id'].to_list()
#sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek'] #TVA
site_prev = None

meas_max = []
meas_min = []
meas_std = []

mage_max = []
mage_min = []
mage_std = []

swmf_max = []
swmf_min = []
swmf_std = []

for idx, row in info_df.iterrows():

    site_id = row['site_id']
    if site_id not in sites:
        continue

    site_data, msk_site_data = read_TVA_or_NERC(row)
    msk_site_data = np.abs(msk_site_data)
    

    # finding number of nans masked
    bad_1 = np.sum(msk_site_data.mask)

    # finding variance
    std_1 = np.std(msk_site_data)

    #finding max
    max_1 = np.max(msk_site_data)

    #finding min
    min_1 = np.min(msk_site_data)

    if row['data_source'] == 'NERC' or row['data_source'] == 'TVA':
        meas_max.append(max_1)
        meas_min.append(min_1)
        meas_std.append(std_1)
    elif row['data_source'] == 'MAGE':
        mage_max.append(max_1)
        mage_min.append(min_1)
        mage_std.append(std_1)
    elif row['data_source'] == 'SWMF':
        swmf_max.append(max_1)
        swmf_min.append(min_1)
        swmf_std.append(std_1)

print("Measured Data:")
print(f"Average Max: {np.mean(meas_max)}")
print(f"Average Min: {np.mean(meas_min)}")
print(f"Average Std: {np.mean(meas_std)}")

print("\nMAGE Data:")
print(f"Average Max: {np.mean(mage_max)}")
print(f"Average Min: {np.mean(mage_min)}")
print(f"Average Std: {np.mean(mage_std)}")

print("\nSWMF Data:")
print(f"Average Max: {np.mean(swmf_max)}")
print(f"Average Min: {np.mean(swmf_min)}")
print(f"Average Std: {np.mean(swmf_std)}")





import os
import pickle

import numpy as np
import pandas as pd
import numpy.ma as ma
import datetime
import json
import time


tva_results = False #print results for TVA gic analysis
gmu_results = True #print results for GMU gic analysis
b_results = False #print results for B analysis
cc_results = False #print results for cc analysis

def subset(time, data, start, stop):
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

start = datetime.datetime(2024, 5, 10, 15, 0)
stop = datetime.datetime(2024, 5, 12, 6, 0)

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

data_dir = os.path.join('..', '2024-May-Storm-data', '_processed')
out_dir = os.path.join('..', '2024-May-Storm-data', '_results')

data_dir = os.path.join('..', '2024-May-Storm-data')
all_dir  = os.path.join(data_dir, '_all')
all_file = os.path.join(all_dir, 'all.pkl')

fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

if cc_results:
    results_dir = os.path.join('..', '2024-May-Storm-data', '_results')

    pkl_file = os.path.join(results_dir, 'cc.pkl')
    print(f"Reading {pkl_file}")
    with open(pkl_file, 'rb') as file:
        df = pickle.load(file)

    print(df.columns)
    print('Max, Min, Mean dist:', df['dist(km)'].max(), df[df['dist(km)']>0]['dist(km)'].min(), df['dist(km)'].mean())
    print('Max, Min, Mean lat:', np.abs(df['lat_diff']).max(), np.abs(df[df['dist(km)']>0]['lat_diff']).min(), np.abs(df['lat_diff']).mean())
    print('Max, Min, Mean std:', df['std_1'].max(), df['std_1'].min(), df['std_1'].mean())
    avg_std = (df['std_1']+df['std_2'])/2
    print('Max, Min, Mean avg std:', avg_std.max(), avg_std.min(), avg_std.mean())
    print('Max, Min, Mean beta diff:', np.abs(df['beta_diff']).max(), np.abs(df[df['dist(km)']>0]['beta_diff']).min(), np.abs(df['beta_diff']).mean())
    print('Max, Min, Mean volt diff:', np.abs(df['volt_diff(kV)']).max(), np.abs(df[df['dist(km)']>0]['volt_diff(kV)']).min(), np.abs(df['volt_diff(kV)']).mean())
    no_volt_diff = df['volt_diff(kV)'].isna().sum()
    zero_volt_diff = (df['volt_diff(kV)'] == 0).sum()
    print(f"Number of pairs with no volt_diff(kV): {no_volt_diff}")
    print(f"Number of pairs with volt_diff(kV) = 0: {zero_volt_diff}")


# Filter out sites with error message
# Also remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[~info_df['error'].str.contains('', na=False)]
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)
#print(info_df)

sites = info_df['site_id'].to_list()
#sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek'] #TVA

info_dict, info_df_all, data_all, plot_info = read(all_file)

if b_results:
    meas_max = []
    meas_min = []
    meas_std = []

    mage_max = []
    mage_min = []
    mage_std = []

    swmf_max = []
    swmf_min = []
    swmf_std = []

    cc_mage_mean = []
    pe_mage_mean = []

    cc_swmf_mean = []
    pe_swmf_mean = []

    for idx, row in info_df.iterrows():

        sid = row['site_id']
        if sid not in sites:
            continue
        if 'B' in row['data_type']:
            mag_types = info_dict[sid]['B'].keys()
        else:
            continue
        if 'calculated' in mag_types and 'measured' in mag_types:
            time_meas = data_all[sid]['B']['measured'][0]['modified']['time']
            data_meas = data_all[sid]['B']['measured'][0]['modified']['data']
            time_meas, data_meas = subset(time_meas, data_meas, start, stop)
            data_meas = np.linalg.norm(data_meas, axis=1)

            max_B = np.max(data_meas)
            min_B = np.min(data_meas)
            std_B = np.std(data_meas)

            meas_max.append(max_B)
            meas_min.append(min_B)
            meas_std.append(std_B)
            for idx, data_source in enumerate(info_dict[sid]['B']['calculated']):
                time_calc = data_all[sid]['B']['calculated'][idx]['original']['time']
                data_calc = data_all[sid]['B']['calculated'][idx]['original']['data']
                time_calc, data_calc = subset(time_calc, data_calc, start, stop)
                data_calc = np.linalg.norm(data_calc[:,0:2], axis=1)
                
                max_1 = np.nanmax(data_calc)
                min_1 = np.nanmin(data_calc)
                std_1 = np.nanstd(data_calc)

                time_meas_ts = [time.mktime(t.timetuple()) for t in time_meas]
                time_calcs_ts = np.array( [time.mktime(t.timetuple()) for t in time_calc] )
                time_calcs_ts = time_calcs_ts[~np.isnan(data_calc)]
                data_calc = data_calc[~np.isnan(data_calc)]
                data_interp = np.interp( time_calcs_ts, time_meas_ts, data_meas )
                
                cc = np.corrcoef(data_interp, data_calc)[0,1]
                
                numer = np.sum((data_interp-data_calc)**2)
                denom = np.sum((data_interp-data_interp[idx].mean())**2)
                pe = 1-numer/denom 
                
                if 'MAGE' == data_source.upper():
                        mage_max.append(max_1)
                        mage_min.append(min_1)
                        mage_std.append(std_1)
                        cc_mage_mean.append(cc)
                        pe_mage_mean.append(pe)
                if 'SWMF' == data_source.upper():
                        swmf_max.append(max_1)
                        swmf_min.append(min_1)
                        swmf_std.append(std_1)
                        cc_swmf_mean.append(cc)
                        pe_swmf_mean.append(pe)


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

    print("\nMAGE CC/PE:")
    print(f"Average cc: {np.mean(cc_mage_mean)}")
    print(f"Average pe: {np.mean(pe_mage_mean)}")   

    print("\nSWMF CC/PE:")
    print(f"Average cc: {np.mean(cc_swmf_mean)}")
    print(f"Average pe: {np.mean(pe_swmf_mean)}") 

if tva_results:
    sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']

    for sid in sites:
        print(f'TVA {sid}:')
        time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
        data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
        time_meas, data_meas = subset(time_meas, data_meas, start, stop)
        print(f'    Measured max: {np.max(data_meas)} A')
        print(f'    Measured min: {np.min(data_meas)} A')
        print(f'    Measured std: {np.std(data_meas)} A')

        for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
            if data_source == 'TVA':
                time_calc = data_all[sid]['GIC']['calculated'][idx]['original']['time']
                data_calc = data_all[sid]['GIC']['calculated'][idx]['original']['data']
                time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])
                # TODO: Document why this is necessary
                data_calc = -data_calc
        print(f'    Calculated max: {np.max(data_calc)} A')
        print(f'    Calculated min: {np.min(data_calc)} A')
        print(f'    Calculated std: {np.std(data_calc)} A')

        data_diff = data_meas-data_calc
        print(f'    Difference max: {np.max(data_diff)} A')
        print(f'    Difference min: {np.min(data_diff)} A')
        print(f'    Difference std: {np.std(data_diff)} A')


if gmu_results:
    TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']
    ccs = []
    pes = []
    number_gmu_sites = 0
    for sid in info_dict.keys():
        if 'GIC' in info_dict[sid].keys():
            gic_types = info_dict[sid]['GIC'].keys()
            if 'measured' and 'calculated' in gic_types:
                time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
                data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
                time_meas, data_meas = subset(time_meas, data_meas, start, stop)

                for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
                    if data_source == 'GMU':
                        time_calc = data_all[sid]['GIC']['calculated'][idx]['original']['time']
                        data_calc = data_all[sid]['GIC']['calculated'][idx]['original']['data'][:, 0:1]
                        data_calc = np.array(data_calc).flatten()
                        time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])

                        cc = np.corrcoef(data_meas, data_calc)
                        if cc[0,1] < 0:
                            data_calc = -data_calc
                        number_gmu_sites += 1
                        ccs.append(np.corrcoef(data_meas, data_calc)[0,1])
                        numer = np.sum((data_meas-data_calc)**2)
                        denom = np.sum((data_meas-data_meas.mean())**2)
                        pes.append(1-numer/denom)
                if sid in TVA_sites:
                    print(f'GMU {sid}:')

                    print(f'    Measured max: {np.max(data_meas)} A')
                    print(f'    Measured min: {np.min(data_meas)} A')
                    print(f'    Measured std: {np.std(data_meas)} A')
                
                    print(f'    Calculated max: {np.max(data_calc)} A')
                    print(f'    Calculated min: {np.min(data_calc)} A')
                    print(f'    Calculated std: {np.std(data_calc)} A')

                    data_diff = data_meas-data_calc
                    print(f'    Difference max: {np.max(data_diff)} A')
                    print(f'    Difference min: {np.min(data_diff)} A')
                    print(f'    Difference std: {np.std(data_diff)} A')

    print('GMU results:')
    num_nan_ccs = sum(np.isnan(cc) for cc in ccs)
    num_nan_pes = sum(np.isnan(pe) for pe in pes)
    ccs = [cc for cc in ccs if not np.isnan(cc)]
    pes = [pe for pe in pes if not np.isnan(pe)]

    print(f'    Number of GMU sites: {number_gmu_sites}')
    print(f'    cc range: {np.min(ccs)} to {np.max(ccs)}')
    print(f'    pe range: {np.min(pes)} to {np.max(pes)}')
    print(f'    Number of NaN cc values: {num_nan_ccs}')
    print(f'    Number of NaN pe values: {num_nan_pes}')








###########################################################################################################
# this is an old function that maybe will be useful at some point but not rn #

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
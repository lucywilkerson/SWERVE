import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

data_dir = os.path.join('..', '..', '2024-AGU-data')
data_dir_gic = os.path.join(data_dir, 'tva', 'gic')

## function to save figure(s)
def plot_save(fname):
  print(f"Saving {fname}_hist")
  plt.savefig(f'{fname}_hist.png',dpi=300)
  plt.savefig(f'{fname}_hist.pdf')
  plt.savefig(f'{fname}_hist.svg')

## defining TVA data, I know we have the info.json but this organization felt more right to me?
TVA_dat={
  "bullrun": {
    "name": "Bull Run",
    "files_calc": ["20240510_BullRunXfrmGIC.dat", "20240511_BullRunXfrmGIC.dat","20240512_BullRunXfrmGIC.dat"],
    "start_calc": ["2024-05-10 00:00:00","2024-05-11 00:00:00","2024-05-12 00:00:00"],
    "files_meas": ["gic-bullrun_20240510.csv"],
    "geo_lat": 36.0,
    "geo_lon": -84.2,  
    "type": "gic",
  },
  "montgomery": {
    "name": "Montgomery",
    "files_calc": ["20240510_MontgomeryGIC.dat", "20240511_MontgomeryGIC.dat", "20240512_MontgomeryGIC.dat"],
    "start_calc": ["2024-05-10 00:00:00","2024-05-11 00:00:00", "2024-05-12 00:00:00"],
    "files_meas": ["gic-montgomery_20240510.csv"],
    "geo_lat": 36.6,
    "geo_lon": -87.3,
    "type": "gic"  
  },
  "union": {
    "name": "Union",
    "files_calc": ["20240510_UnionGIC.dat", "20240511_UnionGIC.dat", "20240512_UnionGIC.dat"],
    "start_calc": ["2024-05-10 00:00:00","2024-05-11 00:00:00", "2024-05-12 00:00:00"],
    "files_meas": ["gic-union_20240510.csv"],
    "geo_lat": 36.4,
    "geo_lon": -83.9,
    "type": "gic"  
  },
  "widowscreek": {
    "name": "Widows Creek",
    "files_calc": ["20240510_WidowsCreek2GIC.dat", "20240511_WidowsCreek2GIC.dat", "20240512_WidowsCreek2GIC.dat"],
    "start_calc": ["2024-05-10 00:00:00","2024-05-11 00:00:00", "2024-05-12 00:00:00"],
    "files_meas": ["gic-widowscreek2_20240510.csv"],
    "geo_lat": 34.9,
    "geo_lon": -85.8,
    "type": "gic"  
  }}


## function to plot measured/calculated GIC and corresponding histogram
## it also reads in the data (this can be made two different functions later if that makes more sense)
def plot_hist(loc):
    # setting bin width for histograms
    w=.5 
    # setting x-limit for histogram
    hist_lim=40 
    # empty arrays to store measured values
    t_m=[]
    gic_m=[]
    # reading measured data
    print(f"Reading {loc["files_meas"][0]}")
    with open(loc["files_meas"][0],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            t_m.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')) #adding to time array
            gic_m.append(float(row[1])) #adding to gic array
    # empty arrays to store calculated values
    t_c=[]
    gic_c=[]
    # reading calculated data
    for idx, file in enumerate(loc["files_calc"]):
        start = loc["start_calc"][idx]
        dto = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        print(f"Reading {file}")
        #file = os.path.join(data_dir, file)
        d, times = np.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
        gic_c.append(d)
        for t in times:
            t_c.append(dto + datetime.timedelta(seconds=t))
    gic_c = np.array(gic_c).flatten()
    t_c = np.array(t_c).flatten()
    # averaging measured values
    gic_avg=[]
    gic_min=[]
    for i in np.arange(len(gic_m)):
        gic_min.append(gic_m[i])
        if i in np.arange(59,len(gic_m),60):
            gic_avg.append(np.mean(gic_min))
            gic_min=[]
        else:
            continue
    # finding difference
    gic_diff=[]
    for i in np.arange(len(gic_c)):
        diff=gic_avg[i]-gic_c[i]
        gic_diff.append(diff)
    # plotting!!
    fig, axs = plt.subplots(2, 1,figsize=(25,10))
    # GIC plot
    axs[0].plot(t_m,gic_m,color='gray',linewidth=.5,label='Measured GIC') # og measured data
    axs[0].plot(t_c,gic_avg,color='k',linewidth=.5,label='1-min Avg Measured GIC') # avg'ed measured data
    axs[0].plot(t_c,gic_c,color='b',linewidth=.5,label='Calculated GIC') #og calculated data
    axs[0].set_xlabel('time') 
    axs[0].set_ylabel('GIC [A]') 
    axs[0].grid()
    axs[0].set_title(loc["name"],fontsize=15) 
    axs[0].legend() 
    # histogram plot
    axs[1].hist(gic_diff,bins=np.arange(min(gic_diff),max(gic_diff)+w,w),density=True)
    axs[1].set_xlabel('Difference (measured-calculated) for GIC')
    axs[1].set_ylabel('probability')
    axs[1].set_yscale('log')
    axs[1].set_xlim([-hist_lim,hist_lim])
    axs[1].grid(linestyle='--')
    # saving plot
    name = loc["name"]
    plot_save(name)
    return fig

# actually making plots
plot_hist(TVA_dat["bullrun"])
plot_hist(TVA_dat["union"])
plot_hist(TVA_dat["widowscreek"])
## you might notice that montgomery is missing
## that's because the measured data is missing 2 seconds somewhere, which throws everything off
## ideally this would be fixed by looping over time instead of length to get the averages but I haven't gotten there yet

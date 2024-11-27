import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as dates

data_dir      = os.path.join('..', '..', '2024-AGU-data')
data_dir_swmf = os.path.join( data_dir, 'SWMF')
data_dir_tva  = os.path.join(data_dir, 'tva', 'mag')

# info block with names of files for analysis
info = {
        "tva":  [ 'bullrun_mag_20240509.pkl', 'union_mag_20240509.pkl'],
        "gap":  [ 'dB_bs_gap-BullRun.pkl', 'dB_bs_gap-Union.pkl'],
        'iono': [ 'dB_bs_iono-BullRun.pkl', 'dB_bs_iono-Union.pkl'],
        'msph': [ 'dB_bs_msph-BullRun.pkl', 'dB_bs_msph-Union.pkl'],
        'title':[ 'Bull Run', 'Union']
      }

from tqdm import tqdm

def readTVA( i, info):
    """ Read the TVA data
    
    inputs:
        i = which data in info block lists to use.
        
        info = info block defined above with the file names used in calculations
        
        data_dir = data directory
    
    outputs:
        dftva = dataframe with TVA data    
    """
    
    filetva = info['tva'][i]
    dftva = pd.read_pickle( filetva )
    
    # Pull data from TVA files. Note, we have total magnetic field from TVA,
    # but SWMF has only changes (delta B).  So we substract the mean B field
    # from the TVA data
    Bxmean = dftva['Bx'].mean()
    Bymean = dftva['By'].mean()
    Bzmean = dftva['Bz'].mean()
    
    dftva['dBx'] = dftva['Bx'] - Bxmean
    dftva['dBy'] = dftva['By'] - Bymean
    dftva['dBz'] = dftva['Bz'] - Bzmean
    
    # Since we're not sure of the magnetometer coordinates, we use B horizontal
    # Determine magnitude of B horizontal
    dftva['dBh'] = np.sqrt( (dftva['Bx'] - Bxmean)**2 + (dftva['By'] - Bymean)**2 )  
    
    return dftva
    
def readSWMF( i, info):
    """ Read the SWMF data
    
    inputs:
        i = which data in info block lists to use.
        
        info = info block defined above with the file names used in calculations
        
        data_dir = data directory
    
    outputs:
        dftot = dataframe with SWMF data    
    """
     
    filegap  = info['gap' ][i]
    fileiono = info['iono'][i]
    filemsph = info['msph'][i]
    
    dfgap  = pd.read_pickle( filegap )
    dfiono = pd.read_pickle( fileiono )
    dfmsph = pd.read_pickle( filemsph )
    
    dftot = pd.DataFrame()
    
    # Pull data from SWMF files. Note, total delta B is sum of gap, ionosphere Hall,
    # ionosphere Pedersen, and magnetospheric contributions
    dftot['Time'] = dfgap['Datetime']
    dftot['Bn'] = dfgap['Bn'] + dfiono['Bnh'] + dfiono['Bnp'] + dfmsph['Bn']
    dftot['Be'] = dfgap['Be'] + dfiono['Beh'] + dfiono['Bnp'] + dfmsph['Be']
    dftot['Bd'] = dfgap['Bd'] + dfiono['Bdh'] + dfiono['Bnp'] + dfmsph['Bd']
    
    # Determine magnitude of delta B horizontal
    dftot['dBh'] = np.sqrt( dftot['Bn']**2 + dftot['Be']**2 )
    
    return dftot

#save fig
def plot_save(fname):
  print(f"Saving {fname}_SWMF")
  plt.savefig(f'{fname}_SWMF.png',dpi=300)
  plt.savefig(f'{fname}_SWMF.pdf')
  plt.savefig(f'{fname}_SWMF.svg')

if __name__ == "__main__":
    # setting bin width for histograms
    w=.5 
    # setting x-limit for histogram
    hist_max=400
    hist_min=-150
    # Loop through entries in info block
    # Generate plots comparing delta B_H from TVA magnetometers and SWMF simulation
    
    for i in range(len(info["tva"])):
        dftva = readTVA( i, info )
        dfswmf = readSWMF( i, info )

        mag_avg=[]
        for t in tqdm(range(len(dfswmf['dBh']))):
            t_start=dfswmf['Time'].iloc[t]
            if t == len(dfswmf['dBh'])-1:
                t_stop = dfswmf['Time'].iloc[len(dfswmf['Time'])-1] + datetime.timedelta(minutes=1)
            else:
                t_stop=dfswmf['Time'].iloc[t+1]
            mag_min = dftva['dBh'][(dftva['Time'] >= t_start) & (dftva['Time'] < t_stop)]
            mag_avg.append(mag_min.mean())
        
        # finding difference
        mag_diff=mag_avg-dfswmf['dBh']
            
        #plotting
        fig, ax = plt.subplots(2, 1,figsize=(12, 6))
        ax[0].plot( dftva['Time'], dftva['dBh'] , color='gray', linewidth=.5, label='TVA measured')
        ax[0].plot( dfswmf['Time'], mag_avg, color='k', linewidth=.5, label='1-min avg TVA measured')
        ax[0].plot( dfswmf['Time'], dfswmf['dBh'], color='b', linewidth=.5, label='SWMF calculated')
        ax[0].set_ylabel(r'$|\delta B_H|$ [nT]')
        ax[0].set_xlabel('Time')
        ax[0].set_title( info['title'][i] )
        ax[0].legend()
        ax[0].grid(linestyle='--')
        date_form = dates.DateFormatter("%m-%d\n%H:%M")
        ax[0].xaxis.set_major_formatter(date_form)
        # histogram plot
        ax[1].hist(mag_diff,bins=np.arange(min(mag_diff),max(mag_diff)+w,w),density=True)
        ax[1].set_xlabel('Difference (measured-calculated) for dB')
        ax[1].set_ylabel('probability')
        ax[1].set_yscale('log')
        ax[1].set_xlim([hist_min,hist_max])
        ax[1].grid(linestyle='--')
        # saving
        fname = info['title'][i]
        plot_save(fname)
        
# repeating for vertical direction
#save fig
def plot_save(fname):
  print(f"Saving {fname}_SWMF_z")
  plt.savefig(f'{fname}_SWMF_z.png',dpi=300)
  plt.savefig(f'{fname}_SWMF_z.pdf')
  plt.savefig(f'{fname}_SWMF_z.svg')

if __name__ == "__main__":
    # setting bin width for histograms
    w=.5 
    # setting x-limit for histogram
    hist_max=400
    hist_min=-hist_max
    # Loop through entries in info block
    # Generate plots comparing delta B_H from TVA magnetometers and SWMF simulation

    
    for i in range(len(info["tva"])):
        dftva = readTVA( i, info )
        dfswmf = readSWMF( i, info )

        mag_avg=[]
        for t in tqdm(range(len(dfswmf['Bd']))):
            t_start=dfswmf['Time'].iloc[t]
            if t == len(dfswmf['Bd'])-1:
                t_stop = dfswmf['Time'].iloc[len(dfswmf['Time'])-1] + datetime.timedelta(minutes=1)
            else:
                t_stop=dfswmf['Time'].iloc[t+1]
            mag_min = dftva['dBz'][(dftva['Time'] >= t_start) & (dftva['Time'] < t_stop)]
            mag_avg.append(mag_min.mean())
        
        # finding difference
        mag_diff=mag_avg-dfswmf['Bd']
            
        #plotting
        fig, ax = plt.subplots(2, 1,figsize=(12, 6))
        ax[0].plot( dftva['Time'], dftva['dBz'] , color='gray', linewidth=.5, label='TVA measured')
        ax[0].plot( dfswmf['Time'], mag_avg, color='k', linewidth=.5, label='1-min avg TVA measured')
        ax[0].plot( dfswmf['Time'], dfswmf['Bd'], color='b', linewidth=.5, label='SWMF calculated')
        ax[0].set_ylabel(r'$\delta B_z$ [nT]')
        ax[0].set_xlabel('Time')
        ax[0].set_title( info['title'][i] )
        ax[0].legend()
        ax[0].grid(linestyle='--')
        date_form = dates.DateFormatter("%m-%d\n%H:%M")
        ax[0].xaxis.set_major_formatter(date_form)
        # histogram plot
        ax[1].hist(mag_diff,bins=np.arange(min(mag_diff),max(mag_diff)+w,w),density=True)
        ax[1].set_xlabel('Difference (measured-calculated) for dB')
        ax[1].set_ylabel('probability')
        ax[1].set_yscale('log')
        ax[1].set_xlim([hist_min,hist_max])
        ax[1].grid(linestyle='--')
        # saving
        fname = info['title'][i]
        plot_save(fname)
        




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime as dt
from scipy import signal
from scipy.optimize import curve_fit
import sys
import os
from swerve import cli, config

CONFIG = config()
data_dir = CONFIG['dirs']['data']

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fitplot(histdata,errrange,binns,lab,ls):
    hist, bin_edges = np.histogram(histdata,range=errrange,bins=binns)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [np.nanmax(hist),np.nanmean(histdata),np.nanstd(histdata)]   
    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)    
    # Get the fitted curve
    xfit=np.linspace(errrange[0],errrange[1],num=500)
    hist_fit = gauss(xfit,*coeff)    
    #plt.plot(bin_centres, hist, label='Test data')
    plt.plot(xfit,hist_fit,'k', alpha=0.5, label=lab,linestyle=ls)
    return coeff[1],coeff[2] #mean,std


def read_gic_file(filepath):
    gic = pd.read_csv(filepath,
                    header=0,
                    na_values='99999.0',
                    usecols=(0,1),
                    parse_dates={'Timestamp':[0]},
                    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                    index_col='Timestamp')
    gic.columns = ['gic']
    gic.sort_index(inplace=True)
    return gic.dropna()


def compute_signature_metrics(series, diff_window=5, std_window=20):
    series = series.squeeze()
    ddat = series.diff() #d/dt proxy
    dddat = ddat.diff().shift(-1) #d^2/dt^2 proxy, shifted to align with the middle point of the 3-point stencil
    d = abs(ddat).rolling(diff_window, center=True).sum() #magnitude of d/dt over a window
    dd = abs(dddat).rolling(diff_window, center=True).sum() #magnitude of d^2/dt^2 over a window
    d_std = abs(series).rolling(std_window, center=True).std() #local variability, helps distinguish spikes from real variation
    return d, dd, d_std


def compute_signature_mask(series, d_thresh, dd_thresh, std_thresh, diff_window=5, std_window=20):
    # find sudden (dd), sharp (d) changes with low background (d_std), i.e. spikes, using thresholds
    d, dd, d_std = compute_signature_metrics(series, diff_window=diff_window, std_window=std_window)
    return (d > d_thresh) & (dd >= dd_thresh) & (d_std <= std_thresh), d, dd, d_std


def expand_mask(mask, pre=2, post=8):
    # expands mask to include values before/after detected spike
    mask = mask.squeeze() if hasattr(mask, 'squeeze') else mask
    broad_mask = np.zeros(len(mask), dtype=bool)
    for i, value in enumerate(mask):
        if value:
            if i < pre:
                broad_mask[:i+post] = True
            elif i > len(mask) - post - 1:
                broad_mask[i-pre:] = True
            else:
                broad_mask[i-pre:i+post] = True
    return broad_mask


def interpolate_gic(cln_gic, mask, method='pchip', limit=60, limit_area='inside'):
    cln_gic = cln_gic.copy()
    cln_gic.loc[mask, 'gic'] = np.nan
    return cln_gic.interpolate(method=method, limit=limit, limit_area=limit_area)


def plot_gic_with_symh(cln_gic, symh, span_len=None, label='SYM-H [nT]'):
    plt.figure()
    plt.plot(cln_gic.index, cln_gic.gic.values)
    ax2 = plt.twinx()
    ax2.plot(symh.index, symh.values, color='k', alpha=0.5)
    if span_len is not None and span_len > 0:
        plt.axvspan(cln_gic.index[0], cln_gic.index[min(span_len, len(cln_gic)-1)], alpha=0.3)
    ax2.set_ylabel(label)
    return ax2


def filter_and_remove_offset(cln_gic, offset_samples, cutoff=1./(24.0*60.0*60), fs=0.5):
    cln_gic = cln_gic.copy()
    filtered = butter_highpass_filter(cln_gic.gic.values, cutoff, fs)
    cln_gic.gic = filtered
    offset_value = cln_gic.iloc[:offset_samples].gic.mean()
    print('Need to correct for offset of', offset_value)
    return cln_gic - offset_value

"""

odata2014=pd.read_csv('/home/mheyns/SANSA/Research/Colab_paper/omni/omni_min2014.nan',
                      header=0,
                      delim_whitespace=True,
                      usecols=(0,1,2,3,41),
                      names=['yr','dy','hr','mn','sh'],
                      parse_dates={'Timestamp':['yr','dy','hr','mn']},
                      date_parser=lambda x: pd.to_datetime(x, format='%Y %j %H %M'),
                      index_col='Timestamp')

odata2015=pd.read_csv('/home/mheyns/SANSA/Research/Colab_paper/omni/omni_min2015.nan',
                      header=0,
                      delim_whitespace=True,
                      usecols=(0,1,2,3,41),
                      names=['yr','dy','hr','mn','sh'],
                      parse_dates={'Timestamp':['yr','dy','hr','mn']},
                      date_parser=lambda x: pd.to_datetime(x, format='%Y %j %H %M'),
                      index_col='Timestamp')

odata2016=pd.read_csv('/home/mheyns/SANSA/Research/Colab_paper/omni/omni_min2016.nan',
                      header=0,
                      delim_whitespace=True,
                      usecols=(0,1,2,3,41),
                      names=['yr','dy','hr','mn','sh'],
                      parse_dates={'Timestamp':['yr','dy','hr','mn']},
                      date_parser=lambda x: pd.to_datetime(x, format='%Y %j %H %M'),
                      index_col='Timestamp')

odata=pd.concat([odata2014,odata2015,odata2016])
odata.dropna(inplace=True)

"""

#%%
file_name = os.path.join(data_dir,'data_original','tva','gic','GIC-original','weigel','montgomery_event_201503.csv')
mont0 = read_gic_file(file_name)
cln_gic = mont0.copy()

mask2, d, dd, d_std = compute_signature_mask(cln_gic.gic, d_thresh=1.0, dd_thresh=10.0, std_thresh=1.0)
broad_mask2 = expand_mask(mask2, pre=2, post=8)

fig,(ax,ax2) = plt.subplots(2,1,sharex=True)
ax.plot(mont0.index, mont0.gic.values, color='b')
ax.plot(mont0[broad_mask2].index, mont0[broad_mask2].gic.values, linestyle='', marker='.', color='r')
ax2.plot(d.index, d.values, linestyle='', marker='.', color='k')
ax2.plot(dd.index, dd.values, linestyle='', marker='.', color='m')
ax2.plot(d_std.index, d_std.values, linestyle='', marker='.', color='g')

cln_gic = interpolate_gic(cln_gic, broad_mask2)

heyns_gic = os.path.join(data_dir,'data_original','tva','gic','GIC-original','heyns','mont_03_2015.csv')
compare_gic = pd.read_csv(heyns_gic, index_col='Timestamp', parse_dates=True)
compare_gic.index = pd.to_datetime(compare_gic.index)
compare_gic.columns = ['gic']
print(compare_gic.head())
print(cln_gic.head())
print(mont0.head())

plt.figure()
plt.plot(mont0.index, mont0.gic.values, color='k', linewidth=1, label='original GIC')
plt.plot(cln_gic.index, cln_gic.gic.values, color='m', linewidth=1, label='spike removed GIC')

#symh = odata.sh.loc[mont0.index].dropna()
#plot_gic_with_symh(cln_gic, symh, span_len=8500)

cln_gic = filter_and_remove_offset(cln_gic, offset_samples=8500)

plt.plot(cln_gic.index, cln_gic.gic.values, color='c', linewidth=1, label='filtered GIC')
plt.plot(compare_gic.index, compare_gic.gic.values, color='y', linewidth=1, linestyle=':', label='Heyns fully clean GIC')
plt.legend()
plt.grid()


plt.figure()
plt.hist(cln_gic.iloc[:8500].gic.values, bins=100)
plt.axvline(cln_gic.iloc[:8500].gic.mean())

plt.show()

print('difference between clean and Heyns clean:', (cln_gic.gic - compare_gic.gic).abs().mean())
exit()

plt.figure()
plt.plot(cln_gic.index, cln_gic.gic.values)
plt.twinx()
plt.plot(symh.index, symh.values, color='k', alpha=0.5)
plt.axvspan(cln_gic.index[0], cln_gic.index[8500], alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mont_03_2015.csv', na_rep='NaN', float_format='%.2f')

#%%
mont1 = read_gic_file('/home/mheyns/SANSA/Research/Data/TVA/event_201506/montgomery_event_201506.csv')
cln_gic = mont1.copy()

symh = odata.sh.loc[mont1.index].dropna()
plot_gic_with_symh(cln_gic, symh, span_len=10000)

cln_gic = filter_and_remove_offset(cln_gic, offset_samples=10000)

plt.figure()
plt.hist(cln_gic.iloc[:10000].gic.values, bins=100)
plt.axvline(cln_gic.iloc[:10000].gic.mean())

plt.figure()
plt.plot(cln_gic.index, cln_gic.gic.values)
plt.twinx()
plt.plot(symh.index, symh.values, color='k', alpha=0.5)
plt.axvspan(cln_gic.index[0], cln_gic.index[10000], alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mont_06_2015.csv', na_rep='NaN', float_format='%.2f')

#%%

mont2 = read_gic_file('/home/mheyns/SANSA/Research/Data/TVA/event_201409/montgomery_event_201409.csv')
cln_gic = mont2.copy()

mask2, d, dd, d_std = compute_signature_mask(cln_gic.gic, d_thresh=4.0, dd_thresh=5.0, std_thresh=1.6)
broad_mask2 = expand_mask(mask2, pre=2, post=8)

fig,(ax,ax2) = plt.subplots(2,1,sharex=True)
ax.plot(mont2.index, mont2.gic.values, color='b')
ax.plot(mont2[broad_mask2].index, mont2[broad_mask2].gic.values, linestyle='', marker='.', color='r')
ax2.plot(d.index, d.values, linestyle='', marker='.', color='k')
ax2.plot(dd.index, dd.values, linestyle='', marker='.', color='m')
ax2.plot(d_std.index, d_std.values, linestyle='', marker='.', color='g')

cln_gic = interpolate_gic(cln_gic, broad_mask2)

symh = odata.sh.loc[mont2.index].dropna()
plot_gic_with_symh(cln_gic, symh, span_len=4800)

cln_gic = filter_and_remove_offset(cln_gic, offset_samples=4800)

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values, bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

plt.figure()
plt.plot(cln_gic.index, cln_gic.gic.values)
plt.twinx()
plt.plot(symh.index, symh.values, color='k', alpha=0.5)
plt.axvspan(cln_gic.index[0], cln_gic.index[10000], alpha=0.3)
plt.ylabel('SYM-H [nT]')

#cln_gic.to_csv('mont_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%%

mont3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/montgomery_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mont3.columns=['gic']     #verify the header
mont3.sort_index(inplace=True)
mont3=mont3.dropna()

cln_gic=mont3.copy()

ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mont3).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=5.0)&(d_std<=2.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mont3.index,mont3.values,color='b')
ax.plot(mont3[broad_mask2].index,mont3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mont3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mont_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%

mont4=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/montgomery_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mont4.columns=['gic']     #verify the header
mont4.sort_index(inplace=True)
mont4=mont4.dropna()

cln_gic=mont4.copy()

ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mont4).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=5.0)&(d_std<=2.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mont4.index,mont4.values,color='b')
ax.plot(mont4[broad_mask2].index,mont4[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mont4.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mont_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%
weak0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/weakley_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
weak0.columns=['gic']     #verify the header
weak0.sort_index(inplace=True)
weak0=weak0.dropna()

cln_gic=weak0.copy()
    
symh=odata.sh.loc[weak0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8500].mean())

cln_gic=cln_gic-cln_gic.iloc[:8500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('weak_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
weak1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/weakley_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
weak1.columns=['gic']     #verify the header
weak1.sort_index(inplace=True)
weak1=weak1.dropna()

cln_gic=weak1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(weak1).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=5.0)&(d_std<=0.8) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if (i<20000)|(i>40000):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(weak1.index,weak1.values,color='b')
ax.plot(weak1[broad_mask2].index,weak1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[weak1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:10000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:10000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:10000].mean())

cln_gic=cln_gic-cln_gic.iloc[:10000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('weak_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%

weak2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/weakley_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
weak2.columns=['gic']     #verify the header
weak2.sort_index(inplace=True)
weak2=weak2.dropna()

cln_gic=weak2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(weak2).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=5.0)&(d_std<=0.8) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if (i<20000)|(i>40000):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(weak2.index,weak2.values,color='b')
ax.plot(weak2[broad_mask2].index,weak2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[weak2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('weak_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%

weak3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/weakley_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
weak3.columns=['gic']     #verify the header
weak3.sort_index(inplace=True)
weak3=weak3.dropna()

cln_gic=weak3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(weak3).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=5.0)&(d_std<=0.8) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if (i<20000)|(i>40000):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(weak3.index,weak3.values,color='b')
ax.plot(weak3[broad_mask2].index,weak3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[weak3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('weak_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%

epnt=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/eastpoint_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
epnt.columns=['gic']     #verify the header
epnt.sort_index(inplace=True)
epnt=epnt.dropna()

cln_gic=epnt.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(epnt).rolling(60,center=True).std()

mask2=(d>0.4)&(dd>=0.8)&(d_std<=0.1) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(epnt.index,epnt.values,color='b')
ax.plot(epnt[broad_mask2].index,epnt[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[epnt.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:10000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:10000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:10000].mean())

cln_gic=cln_gic-cln_gic.iloc[:10000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('epnt_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%

brad0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/bradley_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
brad0.columns=['gic']     #verify the header
brad0.sort_index(inplace=True)
brad0=brad0.dropna()

cln_gic=brad0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(brad0).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(brad0.index,brad0.values,color='b')
ax.plot(brad0[broad_mask2].index,brad0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[brad0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('brad_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%%

brad1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/bradley_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
brad1.columns=['gic']     #verify the header
brad1.sort_index(inplace=True)
brad1=brad1.dropna()

cln_gic=brad1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(brad1).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(brad1.index,brad1.values,color='b')
ax.plot(brad1[broad_mask2].index,brad1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[brad1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('brad_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
brad2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/bradley_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
brad2.columns=['gic']     #verify the header
brad2.sort_index(inplace=True)
brad2=brad2.dropna()

cln_gic=brad2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(brad2).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(brad2.index,brad2.values,color='b')
ax.plot(brad2[broad_mask2].index,brad2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[brad2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('brad_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
brad3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/bradley_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
brad3.columns=['gic']     #verify the header
brad3.sort_index(inplace=True)
brad3=brad3.dropna()

cln_gic=brad3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(brad3).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(brad3.index,brad3.values,color='b')
ax.plot(brad3[broad_mask2].index,brad3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[brad3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('brad_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
brad4=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/bradley_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
brad4.columns=['gic']     #verify the header
brad4.sort_index(inplace=True)
brad4=brad4.dropna()

cln_gic=brad4.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(brad4).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(brad4.index,brad4.values,color='b')
ax.plot(brad4[broad_mask2].index,brad4[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[brad4.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('brad_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%% 
wcrk0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/widowscreek_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
wcrk0.columns=['gic']     #verify the header
wcrk0.sort_index(inplace=True)
wcrk0=wcrk0.dropna()

cln_gic=wcrk0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(wcrk0).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(wcrk0.index,wcrk0.values,color='b')
ax.plot(wcrk0[broad_mask2].index,wcrk0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[wcrk0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('wcrk_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%% 
wcrk1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/widowscreek_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
wcrk1.columns=['gic']     #verify the header
wcrk1.sort_index(inplace=True)
wcrk1=wcrk1.dropna()

cln_gic=wcrk1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(wcrk1).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(wcrk1.index,wcrk1.values,color='b')
ax.plot(wcrk1[broad_mask2].index,wcrk1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[wcrk1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('wcrk_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
wcrk2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/widowscreek_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
wcrk2.columns=['gic']     #verify the header
wcrk2.sort_index(inplace=True)
wcrk2=wcrk2.dropna()

cln_gic=wcrk2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(wcrk2).rolling(60,center=True).std()

mask2=(d>1.2)&(dd>=1.5)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(wcrk2.index,wcrk2.values,color='b')
ax.plot(wcrk2[broad_mask2].index,wcrk2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[wcrk2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('wcrk_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
wcrk3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/widowscreek_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
wcrk3.columns=['gic']     #verify the header
wcrk3.sort_index(inplace=True)
wcrk3=wcrk3.dropna()

cln_gic=wcrk3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(wcrk3).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(wcrk3.index,wcrk3.values,color='b')
ax.plot(wcrk3[broad_mask2].index,wcrk3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[wcrk3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('wcrk_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
wcrk4=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/widowscreek_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
wcrk4.columns=['gic']     #verify the header
wcrk4.sort_index(inplace=True)
wcrk4=wcrk4.dropna()

cln_gic=wcrk4.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(wcrk4).rolling(60,center=True).std()

mask2=(d>2.0)&(dd>=2.0)&(d_std<=0.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(wcrk4.index,wcrk4.values,color='b')
ax.plot(wcrk4[broad_mask2].index,wcrk4[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[wcrk4.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('wcrk_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%% 
bull0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/bullrun_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
bull0.columns=['gic']     #verify the header
bull0.sort_index(inplace=True)
bull0=bull0.dropna()

cln_gic=bull0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(bull0).rolling(60,center=True).std()

mask2=(d>0.8)&(dd>=1.5)&(d_std<=0.8) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if (i<30000)|(i>60000):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(bull0.index,bull0.values,color='b')
ax.plot(bull0[broad_mask2].index,bull0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[bull0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('bull_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
bull1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/bullrun_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
bull1.columns=['gic']     #verify the header
bull1.sort_index(inplace=True)
bull1=bull1.dropna()

cln_gic=bull1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(bull1).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(bull1.index,bull1.values,color='b')
ax.plot(bull1[broad_mask2].index,bull1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[bull1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('bull_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
bull2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/bullrun_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
bull2.columns=['gic']     #verify the header
bull2.sort_index(inplace=True)
bull2=bull2.dropna()

cln_gic=bull2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(bull2).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(bull2.index,bull2.values,color='b')
ax.plot(bull2[broad_mask2].index,bull2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[bull2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

#filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
#cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('bull_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%% 
rccn0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/raccoon_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rccn0.columns=['gic']     #verify the header
rccn0.sort_index(inplace=True)
rccn0=rccn0.dropna()

cln_gic=rccn0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rccn0).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rccn0.index,rccn0.values,color='b')
ax.plot(rccn0[broad_mask2].index,rccn0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[rccn0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rccn_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%% 
rccn1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/raccoon_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rccn1.columns=['gic']     #verify the header
rccn1.sort_index(inplace=True)
rccn1=rccn1.dropna()

cln_gic=rccn1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rccn1).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=2.0)&(d_std<=0.2) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rccn1.index,rccn1.values,color='b')
ax.plot(rccn1[broad_mask2].index,rccn1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')
ax.axvspan(pd.Timestamp(2015,3,17,3,7,55),pd.Timestamp(2015,3,17,3,9,24))

cln_gic[broad_mask2]=np.nan
cln_gic.loc[pd.Timestamp(2015,3,17,3,7,55):pd.Timestamp(2015,3,17,3,9,24)]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[rccn1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rccn_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
rccn2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/raccoon_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rccn2.columns=['gic']     #verify the header
rccn2.sort_index(inplace=True)
rccn2=rccn2.dropna()

cln_gic=rccn2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rccn2).rolling(60,center=True).std()

mask2=(d>0.5)&(dd>=1.0)&(d_std<=0.05) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rccn2.index,rccn2.values,color='b')
ax.plot(rccn2[broad_mask2].index,rccn2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[rccn2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:10000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:10000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:10000].mean())

cln_gic=cln_gic-cln_gic.iloc[:10000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rccn_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
rfrd0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/rutherford_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rfrd0.columns=['gic']     #verify the header
rfrd0.sort_index(inplace=True)
rfrd0=rfrd0.dropna()

cln_gic=rfrd0.copy()

cln_gic.loc[:pd.Timestamp(2015,6,22,9)]=rfrd0.loc[:pd.Timestamp(2015,6,22,9)].resample('4S').mean()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rfrd0).rolling(60,center=True).std()

mask2=(d>1.0)&(dd>=1.0)&(d_std<=0.15) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rfrd0.index,rfrd0.values,color='b')
ax.plot(rfrd0[broad_mask2].index,rfrd0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[rfrd0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:10000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:10000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:10000].mean())

cln_gic=cln_gic-cln_gic.iloc[:10000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[10000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rfrd_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
rfrd1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/rutherford_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rfrd1.columns=['gic']     #verify the header
rfrd1.sort_index(inplace=True)
rfrd1=rfrd1.dropna()

cln_gic=rfrd1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rfrd1).rolling(60,center=True).std()

mask2=(d>0.8)&(dd>=2.0)&(d_std<=1.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rfrd1.index,rfrd1.values,color='b')
ax.plot(rfrd1[broad_mask2].index,rfrd1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[rfrd1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4800].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4800].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4800].mean())

cln_gic=cln_gic-cln_gic.iloc[:4800].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4800],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rfrd_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%% 
rfrd2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/rutherford_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
rfrd2.columns=['gic']     #verify the header
rfrd2.sort_index(inplace=True)
rfrd2=rfrd2.dropna()

cln_gic=rfrd2.copy()

cln_gic.loc[pd.Timestamp(2016,2,17,11,30):]=rfrd2.loc[pd.Timestamp(2016,2,17,11,30):].resample('8S').mean()
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(rfrd2).rolling(60,center=True).std()

mask2=(d>0.8)&(dd>=2.0)&(d_std<=1.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
        if mask2.iloc[i].values==True:
            if i<2:
                broad_mask2[:i+5]=True
            elif i>len(mask2)-6:
                broad_mask2[i-5:]=True
            else:
                broad_mask2[i-2:i+5]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(rfrd2.index,rfrd2.values,color='b')
ax.plot(rfrd2[broad_mask2].index,rfrd2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')
cln_gic=cln_gic.fillna(method='ffill')

symh=odata.sh.loc[rfrd2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('rfrd_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%
para0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/paradise_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
para0.columns=['gic']     #verify the header
para0.sort_index(inplace=True)
para0=para0.dropna()

cln_gic=para0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(para0).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=10.0)&(d_std<=1.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(para0.index,para0.values,color='b')
ax.plot(para0[broad_mask2].index,para0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[para0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4000].mean())

cln_gic=cln_gic-cln_gic.iloc[:4000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('para_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%%
para1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/paradise_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
para1.columns=['gic']     #verify the header
para1.sort_index(inplace=True)
para1=para1.dropna()

cln_gic=para1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(para1).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=0.5) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(para1.index,para1.values,color='b')
ax.plot(para1[broad_mask2].index,para1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[para1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4000].mean())

cln_gic=cln_gic-cln_gic.iloc[:4000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('para_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
para2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/paradise_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
para2.columns=['gic']     #verify the header
para2.sort_index(inplace=True)
para2=para2.dropna()

cln_gic=para2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(para2).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=0.5) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(para2.index,para2.values,color='b')
ax.plot(para2[broad_mask2].index,para2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[para2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4000].mean())

cln_gic=cln_gic-cln_gic.iloc[:4000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('para_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
para3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/paradise_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
para3.columns=['gic']     #verify the header
para3.sort_index(inplace=True)
para3=para3.dropna()

cln_gic=para3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(para3).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=0.5) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(para3.index,para3.values,color='b')
ax.plot(para3[broad_mask2].index,para3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[para3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[1000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:1000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:1000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:1000].mean())

cln_gic=cln_gic-cln_gic.iloc[:1000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[1000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('para_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%
shel0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/shelby_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shel0.columns=['gic']     #verify the header
shel0.sort_index(inplace=True)
shel0=shel0.dropna()

cln_gic=shel0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shel0).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=1.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shel0.index,shel0.values,color='b')
ax.plot(shel0[broad_mask2].index,shel0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shel0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shel_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
shel1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/shelby_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shel1.columns=['gic']     #verify the header
shel1.sort_index(inplace=True)
shel1=shel1.dropna()

cln_gic=shel1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shel1).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=1.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shel1.index,shel1.values,color='b')
ax.plot(shel1[broad_mask2].index,shel1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shel1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shel_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
shel2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/shelby_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shel2.columns=['gic']     #verify the header
shel2.sort_index(inplace=True)
shel2=shel2.dropna()

cln_gic=shel2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shel2).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=1.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shel2.index,shel2.values,color='b')
ax.plot(shel2[broad_mask2].index,shel2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shel2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shel_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
shel3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/shelby_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shel3.columns=['gic']     #verify the header
shel3.sort_index(inplace=True)
shel3=shel3.dropna()

cln_gic=shel3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shel3).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=10.0)&(d_std<=1.25) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shel3.index,shel3.values,color='b')
ax.plot(shel3[broad_mask2].index,shel3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shel3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shel_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%
shvn0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/southaven_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shvn0.columns=['gic']     #verify the header
shvn0.sort_index(inplace=True)
shvn0=shvn0.dropna()

cln_gic=shvn0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shvn0).rolling(20,center=True).std()

mask2=(d>0.75)&(dd>=1.5)&(d_std<=0.8) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shvn0.index,shvn0.values,color='b')
ax.plot(shvn0[broad_mask2].index,shvn0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shvn0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shvn_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%%
shvn1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/southaven_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shvn1.columns=['gic']     #verify the header
shvn1.sort_index(inplace=True)
shvn1=shvn1.dropna()

cln_gic=shvn1.copy()

cln_gic.loc[pd.Timestamp(2015,3,18,2,27):pd.Timestamp(2015,3,18,3,17)]=cln_gic.loc[pd.Timestamp(2015,3,18,2,27):pd.Timestamp(2015,3,18,3,17)].resample('4S').mean()
cln_gic.loc[pd.Timestamp(2015,3,18,4,17):pd.Timestamp(2015,3,18,4,47)]=cln_gic.loc[pd.Timestamp(2015,3,18,4,17):pd.Timestamp(2015,3,18,4,47)].resample('4S').mean()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shvn1).rolling(20,center=True).std()

mask2=(d>1.5)&(dd>=3.0)&(d_std<=0.4) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shvn1.index,shvn1.values,color='b')
ax.plot(shvn1[broad_mask2].index,shvn1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shvn1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shvn_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
shvn2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/southaven_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shvn2.columns=['gic']     #verify the header
shvn2.sort_index(inplace=True)
shvn2=shvn2.dropna()

cln_gic=shvn2.copy()

cln_gic.loc[pd.Timestamp(2015,6,22,4,52):pd.Timestamp(2015,6,22,4,53)]=cln_gic.loc[pd.Timestamp(2015,6,22,4,52):pd.Timestamp(2015,6,22,4,53)].resample('4S').mean()

cln_gic.loc[pd.Timestamp(2015,6,23,7,6):pd.Timestamp(2015,6,23,8,26)]=cln_gic.loc[pd.Timestamp(2015,6,23,7,6):pd.Timestamp(2015,6,23,8,26)].resample('4S').mean()

cln_gic.loc[pd.Timestamp(2015,6,23,12,8):pd.Timestamp(2015,6,23,14,52)]=cln_gic.loc[pd.Timestamp(2015,6,23,12,8):pd.Timestamp(2015,6,23,14,52)].resample('4S').mean()

cln_gic.loc[pd.Timestamp(2015,6,23,16,9):pd.Timestamp(2015,6,23,17,19)]=cln_gic.loc[pd.Timestamp(2015,6,23,16,9):pd.Timestamp(2015,6,23,17,19)].resample('4S').mean()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shvn2).rolling(20,center=True).std()

mask2=(d>0.75)&(dd>=1.5)&(d_std<=0.125) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shvn2.index,shvn2.values,color='b')
ax.plot(shvn2[broad_mask2].index,shvn2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shvn2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shvn_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
shvn3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201602/southaven_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
shvn3.columns=['gic']     #verify the header
shvn3.sort_index(inplace=True)
shvn3=shvn3.dropna()

cln_gic=shvn3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(shvn3).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=4.0)&(d_std<=0.5) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(shvn3.index,shvn3.values,color='b')
ax.plot(shvn3[broad_mask2].index,shvn3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[shvn3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:500].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:500].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:500].mean())

cln_gic=cln_gic-cln_gic.iloc[:500].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[500],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('shvn_02_2016.csv',na_rep='NaN',float_format='%.2f')

#%%
sull0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201409/sullivan_event_201409.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
sull0.columns=['gic']     #verify the header
sull0.sort_index(inplace=True)
sull0=sull0.dropna()

cln_gic=sull0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(sull0).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=4.0)&(d_std<=3.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(sull0.index,sull0.values,color='b')
ax.plot(sull0[broad_mask2].index,sull0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[sull0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:4000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:4000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:4000].mean())

cln_gic=cln_gic-cln_gic.iloc[:4000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[4000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('sull_09_2014.csv',na_rep='NaN',float_format='%.2f')

#%%
sull1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201503/sullivan_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
sull1.columns=['gic']     #verify the header
sull1.sort_index(inplace=True)
sull1=sull1.dropna()

cln_gic=sull1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(sull1).rolling(20,center=True).std()

mask2=(d>2.0)&(dd>=3.0)&(d_std<=2.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(sull1.index,sull1.values,color='b')
ax.plot(sull1[broad_mask2].index,sull1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[sull1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('sull_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
sull2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201506/sullivan_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
sull2.columns=['gic']     #verify the header
sull2.sort_index(inplace=True)
sull2=sull2.dropna()

cln_gic=sull2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(sull2).rolling(20,center=True).std()

mask2=(d>1.8)&(dd>=3.0)&(d_std<=3.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(sull2.index,sull2.values,color='b')
ax.plot(sull2[broad_mask2].index,sull2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[sull2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('sull_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
sull3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/event_201510/sullivan_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
sull3.columns=['gic']     #verify the header
sull3.sort_index(inplace=True)
sull3=sull3.dropna()

cln_gic=sull3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(sull3).rolling(20,center=True).std()

mask2=(d>1.8)&(dd>=3.0)&(d_std<=3.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(sull3.index,sull3.values,color='b')
ax.plot(sull3[broad_mask2].index,sull3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[sull3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('sull_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
mead0=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/meadowbrook_event_201503.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mead0.columns=['gic']     #verify the header
mead0.sort_index(inplace=True)
mead0=mead0.dropna()

cln_gic=mead0.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mead0).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=10.0)&(d_std<=7.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mead0.index,mead0.values,color='b')
ax.plot(mead0[broad_mask2].index,mead0[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mead0.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mead_03_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
mead1=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/meadowbrook_event_201506.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mead1.columns=['gic']     #verify the header
mead1.sort_index(inplace=True)
mead1=mead1.dropna()

cln_gic=mead1.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mead1).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=8.0)&(d_std<=7.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mead1.index,mead1.values,color='b')
ax.plot(mead1[broad_mask2].index,mead1[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mead1.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

#filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
#cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mead_06_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
mead2=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/meadowbrook_event_201510.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mead2.columns=['gic']     #verify the header
mead2.sort_index(inplace=True)
mead2=mead2.dropna()

cln_gic=mead2.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mead2).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=8.0)&(d_std<=7.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mead2.index,mead2.values,color='b')
ax.plot(mead2[broad_mask2].index,mead2[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mead2.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:8000].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:8000].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:8000].mean())

cln_gic=cln_gic-cln_gic.iloc[:8000].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[8000],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mead_10_2015.csv',na_rep='NaN',float_format='%.2f')

#%%
mead3=pd.read_csv('/home/mheyns/SANSA/Research/Data/TVA/meadowbrook_event_201602.csv',
                  header=0,
                  na_values='99999.0',
                  usecols=(0,1),
                  parse_dates={'Timestamp':[0]}, 
                  date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                  index_col='Timestamp')
mead3.columns=['gic']     #verify the header
mead3.sort_index(inplace=True)
mead3=mead3.dropna()

cln_gic=mead3.copy()
    
ddat=cln_gic.diff()
dddat=ddat.diff().shift(-1)
d=abs(ddat).rolling(5,center=True).sum()
dd=abs(dddat).rolling(5,center=True).sum()
d_std=abs(mead3).rolling(20,center=True).std()

mask2=(d>1.0)&(dd>=8.0)&(d_std<=7.0) #total signature
broad_mask2=np.zeros(len(mask2)).astype('bool')
for i in range(len(mask2)):
    if mask2.iloc[i].values==True:
        if i<2:
            broad_mask2[:i+8]=True
        elif i>len(mask2)-9:
            broad_mask2[i-8:]=True
        else:
            broad_mask2[i-2:i+8]=True

fig,(ax,ax2)=plt.subplots(2,1,sharex=True)
ax.plot(mead3.index,mead3.values,color='b')
ax.plot(mead3[broad_mask2].index,mead3[broad_mask2].values,linestyle='',marker='.',color='r')
ax2.plot(d.index,d.values,linestyle='',marker='.',color='k')
ax2.plot(dd.index,dd.values,linestyle='',marker='.',color='m')
ax2.plot(d_std.index,d_std.values,linestyle='',marker='.',color='g')

cln_gic[broad_mask2]=np.nan
cln_gic=cln_gic.interpolate(method='pchip',limit=60,limit_area='inside')

symh=odata.sh.loc[mead3.index]
symh=symh.dropna()
plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[700],alpha=0.3)
plt.ylabel('SYM-H [nT]')

filtered_sine = butter_highpass_filter(cln_gic.gic.values,1./(24.0*60.0*60),0.5)
cln_gic.gic=filtered_sine

plt.figure()
plt.hist(cln_gic.iloc[:700].gic.values,bins=100)
plt.axvline(cln_gic.iloc[:700].gic.mean())

print('Need to correct for offset of',cln_gic.iloc[:700].mean())

cln_gic=cln_gic-cln_gic.iloc[:700].mean()

plt.figure()
plt.plot(cln_gic.index,cln_gic.values)
plt.twinx()
plt.plot(symh.index,symh.values,color='k',alpha=0.5)
plt.axvspan(cln_gic.index[0],cln_gic.index[700],alpha=0.3)
plt.ylabel('SYM-H [nT]')

cln_gic.to_csv('mead_02_2016.csv',na_rep='NaN',float_format='%.2f')

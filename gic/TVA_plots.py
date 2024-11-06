# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# This notebook reads in both measured and calculated GIC data from TVA for May 10-13, 2024

# %%
#importing necessary modules
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

# %%
#reading in "measured" TVA GIC data 

# importing csv module
import csv

# csv file name
GIC_m = "data/gic/gic-anderson_20240510.csv" #GIC data w/ 1s cadence from May 10-12

# empty arrays to store values
t_m=[]
gic_m=[]

with open(GIC_m,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        t_m.append(row[0]) #adding to time array
        gic_m.append(float(row[1])) #adding to gic array

# %%
#plotting GIC data over 3 day period
plt.figure(figsize=(15,5))
plt.plot(t_m,gic_m,linewidth=.5,color='k',label='Measured GIC') 
plt.xlabel('time') 
plt.ylabel('GIC (Amps)') 
plt.xticks(np.arange(0,259200,43200))
plt.grid()
plt.title('TVA measured GIC') 
plt.legend() 
plt.show() 


# %%
#reading in calculated data - BULL RUN

#data file names - Bull Run
BR10 = 'data/gic/20240510_BullRunXfrmGIC.dat' #GIC calculated data w/ 1m cadence from May 10
BR11 = 'data/gic/20240511_BullRunXfrmGIC.dat' #GIC calculated data w/ 1m cadence from May 11

#defining variables
gic_c10,t_c10=np.loadtxt(BR10,unpack=True,skiprows=1,delimiter=',')
gic_c11,t_c11=np.loadtxt(BR11,unpack=True,skiprows=1,delimiter=',')
t_c11=t_c11+t_c10[1439]

#combining GIC
BR_gic=[]
for i in np.arange(0,1440):
    BR_gic.append(gic_c10[i])
for i in np.arange(0,1440):
    BR_gic.append(gic_c11[i])

#defining t_c using t_m notation :)
t_c=[]
for i in np.arange(0,172800,60):
    t_c.append(t_m[i])

# %%
plt.figure(figsize=(15,5))
plt.plot(t_c,BR_gic,color='b',linewidth=.5)
plt.xlabel('time') 
plt.ylabel('GIC (Amps)') 
plt.xticks(np.arange(0,2880,720))
plt.grid()
plt.title('TVA calculated GIC (Bull Run)') 
#plt.legend() 
plt.show() 

# %%
#reading in calculated data - MONTGOMERY

#data file names - Montgomery
MG10 = 'data/gic/20240510_MontgomeryGIC.dat' #GIC calculated data w/ 1m cadence from May 10
MG11 = 'data/gic/20240511_MontgomeryGIC.dat' #GIC calculated data w/ 1m cadence from May 11

#defining variables
gic_c10,t_c10=np.loadtxt(MG10,unpack=True,skiprows=1,delimiter=',')
gic_c11,t_c11=np.loadtxt(MG11,unpack=True,skiprows=1,delimiter=',')
t_c11=t_c11+t_c10[1439]

#combining GIC
MG_gic=[]
for i in np.arange(0,1440):
    MG_gic.append(gic_c10[i])
for i in np.arange(0,1440):
    MG_gic.append(gic_c11[i])


# %%
plt.figure(figsize=(15,5))
plt.plot(t_c,MG_gic,color='m',linewidth=.5)
plt.xlabel('time') 
plt.ylabel('GIC (Amps)') 
plt.xticks(np.arange(0,2880,720))
plt.grid()
plt.title('TVA calculated GIC (Montgomery)') 
#plt.legend() 
plt.show() 

# %%
#reading in calculated data - UNION

#data file names - Union
UN10 = 'data/gic/20240510_UnionGIC.dat' #GIC calculated data w/ 1m cadence from May 10
UN11 = 'data/gic/20240511_UnionGIC.dat' #GIC calculated data w/ 1m cadence from May 11

#defining variables
gic_c10,t_c10=np.loadtxt(UN10,unpack=True,skiprows=1,delimiter=',')
gic_c11,t_c11=np.loadtxt(UN11,unpack=True,skiprows=1,delimiter=',')
t_c11=t_c11+t_c10[1439]

#combining GIC
UN_gic=[]
for i in np.arange(0,1440):
    UN_gic.append(gic_c10[i])
for i in np.arange(0,1440):
    UN_gic.append(gic_c11[i])

# %%
plt.figure(figsize=(15,5))
plt.plot(t_c,UN_gic,color='g',linewidth=.5)
plt.xlabel('time') 
plt.ylabel('GIC (Amps)') 
plt.xticks(np.arange(0,2880,720))
plt.grid()
plt.title('TVA calculated GIC (Union)') 
#plt.legend() 
plt.show() 

# %%
#reading in calculated data - WIDOWS CREEK

#data file names - Widows Creek
WC10 = 'data/gic/20240510_WidowsCreek2GIC.dat' #GIC calculated data w/ 1m cadence from May 10
WC11 = 'data/gic/20240511_WidowsCreek2GIC.dat' #GIC calculated data w/ 1m cadence from May 11

#defining variables
gic_c10,t_c10=np.loadtxt(WC10,unpack=True,skiprows=1,delimiter=',')
gic_c11,t_c11=np.loadtxt(WC11,unpack=True,skiprows=1,delimiter=',')
t_c11=t_c11+t_c10[1439]

#combining GIC
WC_gic=[]
for i in np.arange(0,1440):
    WC_gic.append(gic_c10[i])
for i in np.arange(0,1440):
    WC_gic.append(gic_c11[i])

# %%
plt.figure(figsize=(15,5))
plt.plot(t_c,WC_gic,color='r',linewidth=.5)
plt.xlabel('time') 
plt.ylabel('GIC (Amps)') 
plt.xticks(np.arange(0,2880,720))
plt.grid()
plt.title('TVA calculated GIC (Widows Creek)') 
#plt.legend() 
plt.show() 

# %%
# 4 panel plot w each station

fig, axs = plt.subplots(2, 2,figsize=(23,10))
fig.suptitle('TVA calculated GIC',fontsize=20)
axs[0, 0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[0, 0].plot(t_c,BR_gic,color='b',linewidth=.5,label='Calculated GIC')
axs[0, 0].set_title('Bull Run',fontsize=15) 
axs[0, 1].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[0, 1].plot(t_c,MG_gic,color='m',linewidth=.5,label='Calculated GIC')
axs[0, 1].set_title('Montgomery',fontsize=15)
axs[1, 0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[1, 0].plot(t_c,UN_gic,color='g',linewidth=.5,label='Calculated GIC')
axs[1, 0].set_title('Union',fontsize=15)
axs[1, 1].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[1, 1].plot(t_c,WC_gic,color='r',linewidth=.5,label='Calculated GIC')
axs[1, 1].set_title('Widows Creek',fontsize=15)

for ax in axs.flat:
    ax.set(xlabel='time', ylabel='GIC (Amps)')
    ax.set_xticks(np.arange(0,172800,43200))
    ax.set_yticks(np.arange(-20,45,10))
    ax.legend()
    ax.grid()

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
#analyzing model agreement - BR

#averaging measured values
gic_avg=[]
gic_min=[]
for i in np.arange(172800):
    gic_min.append(gic_m[i])
    if i in np.arange(59,172800,60):
        gic_avg.append(np.mean(gic_min))
        gic_min=[]
    else:
        continue

#finding difference
BR_diff=[]

for i in np.arange(2880):
    diff=gic_avg[i]-BR_gic[i]
    BR_diff.append(diff)
   



# %%
#GIC plots along w histogram
fig, axs = plt.subplots(2, 1,figsize=(25,10))
fig.suptitle('TVA Bull Run comparison',fontsize=15)
axs[0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') #og measured data
axs[0].plot(t_c,gic_avg[0:2880],color='k',linewidth=.5,label='1-min Avg Measured GIC') #avg'ed measured data
axs[0].plot(t_c,BR_gic,color='b',linewidth=.5,label='Calculated GIC')
axs[0].set_xlabel('time') 
axs[0].set_ylabel('GIC (Amps)') 
axs[0].set_xticks(np.arange(0,172800,43200)) #if plotting og data
#axs[0].set_xticks(np.arange(0,2880,720)) #if only using minute resolution

axs[0].grid()
axs[0].set_title('TVA GIC (Bull Run)') 
axs[0].legend() 

w=.5 #setting bin width for histograms

axs[1].hist(BR_diff,bins=np.arange(min(BR_diff),max(BR_diff)+w,w),density=True)
axs[1].set_xlabel('Difference (measured-calculated) for GIC')
axs[1].set_ylabel('probability')
axs[1].set_xlim([-40,40])
axs[1].set_ylim([0,.7])
axs[1].grid(linestyle='--')
fig.show()

#saving plot
plt.savefig('BR_plot.png',dpi=300)
plt.savefig('BR_plot.svg')
plt.savefig('BR_plot.pdf')

# %%
# analyzing model agreement - MG

#finding difference
MG_diff=[]

for i in np.arange(2880):
    diff=gic_avg[i]-MG_gic[i]
    MG_diff.append(diff)
   

# %%
#GIC plots along w histogram
fig, axs = plt.subplots(2, 1,figsize=(25,10))
fig.suptitle('TVA Montgomery comparison',fontsize=15)
axs[0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[0].plot(t_c,gic_avg[0:2880],color='k',linewidth=.5,label='1-min Avg Measured GIC') #avg'ed measured data
axs[0].plot(t_c,MG_gic,color='m',linewidth=.5,label='Calculated GIC')
axs[0].set_xlabel('time') 
axs[0].set_ylabel('GIC (Amps)') 
axs[0].set_xticks(np.arange(0,172800,43200))
axs[0].grid()
axs[0].set_title('TVA GIC (Montgomery)') 
axs[0].legend() 

axs[1].hist(MG_diff,bins=np.arange(min(MG_diff),max(MG_diff)+w,w),density=True)
axs[1].set_xlabel('Difference (measured-calculated) for GIC')
axs[1].set_ylabel('probability')
axs[1].set_xlim([-40,40])
axs[1].set_ylim([0,.7])
axs[1].grid(linestyle='--')
fig.show()

#saving plot
plt.savefig('MG_plot.png',dpi=300)
plt.savefig('MG_plot.svg')
plt.savefig('MG_plot.pdf')

# %%
# analyzing model agreement - UN

#finding difference
UN_diff=[]

for i in np.arange(2880):
    diff=gic_avg[i]-UN_gic[i]
    UN_diff.append(diff)
   

# %%
#GIC plots along w histogram
fig, axs = plt.subplots(2, 1,figsize=(25,10))
fig.suptitle('TVA Union comparison',fontsize=15)
axs[0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[0].plot(t_c,gic_avg[0:2880],color='k',linewidth=.5,label='1-min Avg Measured GIC') #avg'ed measured data
axs[0].plot(t_c,UN_gic,color='g',linewidth=.5,label='Calculated GIC')
axs[0].set_xlabel('time') 
axs[0].set_ylabel('GIC (Amps)') 
axs[0].set_xticks(np.arange(0,172800,43200))
axs[0].grid()
axs[0].set_title('TVA GIC (Union)') 
axs[0].legend() 

axs[1].hist(UN_diff,bins=np.arange(min(UN_diff),max(UN_diff)+w,w),density=True)
axs[1].set_xlabel('Difference (measured-calculated) for GIC')
axs[1].set_ylabel('probability')
axs[1].set_xlim([-40,40])
axs[1].set_ylim([0,.7])
axs[1].grid(linestyle='--')
fig.show()

#saving plot
plt.savefig('UN_plot.png',dpi=300)
plt.savefig('UN_plot.svg')
plt.savefig('UN_plot.pdf')

# %%
# analyzing model agreement - WC

#finding difference
WC_diff=[]

for i in np.arange(2880):
    diff=gic_avg[i]-WC_gic[i]
    WC_diff.append(diff)
   

# %%
#GIC plots along w histogram
fig, axs = plt.subplots(2, 1,figsize=(25,10))
fig.suptitle('TVA Widows Creek comparison',fontsize=15)
axs[0].plot(t_m[0:172800],gic_m[0:172800],color='gray',linewidth=.5,label='Measured GIC') 
axs[0].plot(t_c,gic_avg[0:2880],color='k',linewidth=.5,label='1-min Avg Measured GIC') #avg'ed measured data
axs[0].plot(t_c,WC_gic,color='r',linewidth=.5,label='Calculated GIC')
axs[0].set_xlabel('time') 
axs[0].set_ylabel('GIC (Amps)') 
axs[0].set_xticks(np.arange(0,172800,43200))
axs[0].grid()
axs[0].set_title('TVA GIC (Widows Creek)') 
axs[0].legend() 


axs[1].hist(WC_diff,bins=np.arange(min(WC_diff),max(WC_diff)+w,w),density=True)
axs[1].set_xlabel('Difference (measured-calculated) for GIC')
axs[1].set_ylabel('probability')
axs[1].set_xlim([-40,40])
axs[1].set_ylim([0,.7])
axs[1].grid(linestyle='--')
fig.show()

#saving plot
plt.savefig('WC_plot.png',dpi=300)
plt.savefig('WC_plot.svg')
plt.savefig('WC_plot.pdf')

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of image file paths
image_paths = ["BR_plot.png", "MG_plot.png", "UN_plot.png", "WC_plot.png"]

# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2,figsize=(75,30))

# Iterate through the image paths and plot them in subplots
for i, path in enumerate(image_paths):
    # Load the image
    img = mpimg.imread(path)

    # Calculate subplot indices
    row = i // 2
    col = i % 2

    # Plot the image in the corresponding subplot
    axes[row, col].imshow(img)
    axes[row, col].axis("off")  # Turn off axis labels

#plt.tight_layout()  # Adjust layout for better spacing

# Adjust spacing
plt.subplots_adjust(wspace=0, hspace=0)

#plt.show()

#saving plot
#plt.savefig('TVA_plot.png',dpi=300)
#plt.savefig('TVA_plot.svg')
#plt.savefig('TVA_plot.pdf')

# %%

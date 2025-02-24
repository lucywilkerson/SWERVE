# 2024-AGU

```
# Send request to Lucy Wilkerson <lwilker@gmu.edu> for access 2024-AGU-data repo
# (it contains NERC data, which is not open)
git clone https://github.com/lucywilkerson/2024-AGU-data
git clone https://github.com/lucywilkerson/2024-AGU
cd 2024-AGU
pip install -e .
```

See stdout for files read and written

```
python info.py
python read.py
<<<<<<< HEAD
python beta.py
python plot_maps.py
python plot_scatter.py
python plot_timeseries.py
python plot_voltage.py
```
=======
```

```
# Read ../2024-AGU-data/_all/all.pkl and info/info.json and plots time series
# of original and modified data in site id subdirectories of
# ../2024-AGU-data/_processed. Also creates and writes plots for sites
# with both measured and calculated data to site id subdirectories of
# ../2024-AGU-data/_processed
python plot.py
```

```
# Read ../2024-AGU-data/_all/all.pkl and write
# ../2024-AGU-data/_results/cc_vs_dist_scatter.png
# ../2024-AGU-data/_results/cc_vs_ave_std_scatter.png
python cc_plot.py
```


>>>>>>> 1602b2080fc16d27fa39762c7619ab387a50de45

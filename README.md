# 2024-AGU

```
# Send request to Lucy Wilkerson <lwilker@gmu.edu> for access 2024-AGU-data repo
# (it contains NERC data, which is not open)
git clone https://github.com/lucywilkerson/2024-AGU-data
git clone https://github.com/lucywilkerson/2024-AGU
cd 2024-AGU
pip install -e .
```

```
# Create info/info.json, which is a dict with keys of site id and values
# containing data types available at each site.
python info.py
```

```
# Create location map and save in ../2024-AGU-data/_map/
python map.py
```

```
# Read all data files in data subdirs of ../2024-AGU-data and write pkl files
# with data for each site as NumPy arrays in site id subdirectories of
# ../2024-AGU-data/_processed. Also write pkl file with all data in
# ../2024-AGU-data/_all/all.pkl.
python read.py
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
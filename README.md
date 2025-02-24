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
python beta.py
python plot_maps.py
python plot_scatter.py
python plot_timeseries.py
python plot_voltage.py
```
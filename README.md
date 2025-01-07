# 2024-AGU

```
# Send request to Lucy Wilkerson <lwilker@gmu.edu> for access 2024-AGU-data repo
# (it contains NERC data, which is not open)
git clone https://github.com/lucywilkerson/2024-AGU-data
git clone https://github.com/lucywilkerson/2024-AGU
cd 2024-AGU
pip install -e .

python info.py # Creates info/info_dict.json and info/info_dataframe.pkl
python map.py
python read.py
python plot.py
```

Note that data from MAGE is preliminary because don't have all the information needed to determine coordinate system of dB. See comment in `read.py`

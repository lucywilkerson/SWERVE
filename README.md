# 2024-AGU

```
# Send request to Lucy Wilkerson <lwilker@gmu.edu> for access 2024-AGU-data repo
# (it contains NERC data, which is not open)
git clone https://github.com/lucywilkerson/2024-AGU-data
git clone https://github.com/lucywilkerson/2024-AGU
cd 2024-AGU
pip install -e .

# Creates info/info.json, which is a dict with keys of site id and values
# containing data types available at each site.
python info.py

python map.py

# Read all data files in ../2024-AGU-data and write pkl files with normalized
# data for each site to ../2024-AGU-data/_processed. Write pkl file with all
# data in ../2024-AGU-data/_all/all.pkl.
python read.py 


python plot.py
```
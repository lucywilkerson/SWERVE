import os
import csv
import json
import pandas as pd

"""
Converts info/info.csv, which has the form

Bull Run,36.0193,-84.1575,GIC,measured,TVA,
Bull Run,36.0193,-84.1575,GIC,calculated,TVA,"error message"
Bull Run,36.0193,-84.1575,GIC,calculated,MAGE,
Bull Run,36.0193,-84.1575,B,measured,TVA,
Bull Run,36.0193,-84.1575,B,calculated,SWMF,
Bull Run,36.0193,-84.1575,B,calculated,MAGE,

to a dict of the form

{
  "Bull Run": {
    "GIC": {
      "measured": "TVA",
      "calculated": ["TVA", "MAGE"]
    },
    "B": {
      "calculated": ["SWMF", "MAGE"]
    }
  }
}

and saves in info/info.json
"""

df = pd.read_csv(os.path.join('info', 'info.csv'))

sites = {}
locations = {}

for idx, row in df.iterrows():
  print(row)
  site, geo_lat, geo_lon, data_type, data_class, data_source, error = row
  if isinstance(error, str) and error.startswith("x "):
    print(f"  Skipping site '{site}' due to error message in info.csv:\n    {error}")
    continue

  locations[site] = (float(geo_lat), float(geo_lon))

  if site not in sites:
    sites[site] = {}
  if data_type not in sites[site]:  # e.g., GIC, B
    sites[site][data_type] = {}
  if data_class not in sites[site][data_type]:
    sites[site][data_type][data_class] = [data_source]
  else:
    sites[site][data_type][data_class].append(data_source)

print("Writing info/info.json")
with open(os.path.join('info','info.json'), 'w') as f:
  json.dump(sites, f, indent=2)

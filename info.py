import os
import csv
import json

"""
Converts info/info.csv, which has the form

Bull Run,36.0193,-84.1575,GIC,measured,TVA
Bull Run,36.0193,-84.1575,GIC,calculated,TVA
Bull Run,36.0193,-84.1575,GIC,calculated,MAGE
Bull Run,36.0193,-84.1575,B,measured,TVA
Bull Run,36.0193,-84.1575,B,calculated,SWMF
Bull Run,36.0193,-84.1575,B,calculated,MAGE

to dicts of the form

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

and

{
  "Bull Run": [36.0193, -84.1575]
}

and saves in info/info_data.json and info/info_locations.json
"""

with open(os.path.join('info','info.csv'), 'r') as f:
  print("Reading info/info.csv")
  rows = csv.reader(f, delimiter=',')
  sites = {}
  locations = {}
  head = next(rows)
  for row in rows:
    # data_type = GIC, B
    # data_class = measured, calculated
    # data_source = TVA, MAGE, SWMF
    site, geo_lat, geo_lon, data_type, data_class, data_source, error = row
    if error != "":
      print(f"  Skipping site '{site}' due to error message in info.csv:\n    {error}")
      continue

    locations[site] = (float(geo_lat), float(geo_lon))

    if site not in sites:
      sites[site] = {}
    if data_type not in sites[site]: # e.g., GIC, B
      sites[site][data_type] = {}
    if data_class not in sites[site][data_type]:
      sites[site][data_type][data_class] = [data_source]
    else:
      sites[site][data_type][data_class].append(data_source)

print("Writing info/info_data.json")
with open(os.path.join('info','info_data.json'), 'w') as f:
  json.dump(sites, f, indent=2)

print("Writing info/info_locations.json")
with open(os.path.join('info','info_locations.json'), 'w') as f:
  json.dump(sites, f, indent=2)

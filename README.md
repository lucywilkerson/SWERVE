**Space Weather Event Response Validation & Evaluation (SWERVE)** 

## Overview
This repository is part of the [ChronoStorm project](https://chronostorm.vercel.app), funded by the U.S. National Science Foundation (NSF) under grant [#2434136](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2434136). The project aims to collect and analyze perishable operational data from critical infrastructure sectors during significant space weather events, particularly during the solar maximum of Solar Cycle 25.

SWERVE provides a centralized environment for validating and comparing space weather metrics for over 75 solar storms between 2003 and 2026, including magnetic field variations and geomagnetically induced currents (GICs). This repository offers validated, cross-referenced data and tools for researchers and infrastructure stakeholders interested in space weather assessment.

## Project Background
In May 2024, Earth experienced a series of extreme space weather events, collectively referred to as the "Gannon Storm." These events included multiple Earth-directed coronal mass ejections (CMEs), triggering intense geomagnetic storms between May 10–12, 2024. 

The storm provided a rare opportunity to observe rapid fluctuations in Earth's magnetic field and to measure the resulting geoelectric fields and GICs, which pose risks to critical infrastructure, especially electrical power systems. SWERVE supports research into how different measurement and modeling approaches quantitatively compare in capturing these phenomena.

## Repository Contents
- **Data Files**: Magnetic field data, geoelectric field estimates, and GIC measurements from various sources.
- **Analysis Scripts**: Code for preprocessing, analyzing, and visualizing the storm response across physical domains.
- **Documentation**: Descriptions of data sources, validation methodology, and results from comparative analyses.

## Goals
- Validate and compare a range of datasets and model outputs for storm-time metrics.
- Provide accessible, structured data for the research community.
- Support space weather infrastructure resilience efforts through a better understanding of space weather impacts.

## Paper Citations
Lucy Wilkerson, Robert S Weigel, Dean Thomas, et al. GIC-Related Observations During the May 2024 Geomagnetic Storm in the United States. _ESS Open Archive_. October 16, 2025. doi: https://doi.org/10.22541/essoar.176062871.15246663/v1

## Funding Acknowledgment
This work was supported by the U.S. National Science Foundation (NSF) under grant [#2434136](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2434136) as part of the ChronoStorm project. The research focuses on collecting and analyzing perishable operational data from critical infrastructure sectors during significant space weather events, aiming to enhance our understanding of infrastructure vulnerabilities and resilience. 

This grant is co-funded by the NSF GEO/AGS Space Weather Research program and the ENG/CMMI Humans, Disasters, and the Built Environment (HDBE) program.

## Data Acknowledgment
We kindly thank the Tennessee Valley Authority (TVA) for providing data from their internal GIC estimation model, reflecting the realistic circuit configuration over the May 10- 12, 2024, period. Without TVA's support, and this perishable data, we would struggle to compare predictive metrics to this more reliable baseline. 

## Contributors 
- Lucy Wilkerson - Graduate Research Assistant (lwilker [ at ] gmu [ dot ] edu) 
- Bob Weigel - Senior Personnel (rweigel [ at ] gmu [ dot ] edu)
- Dean Thomas - Research Fellow (dthomas6 [ at ] gmu [ dot ] edu) 
- Edward Oughton - Primary Investigator (eoughton [ at ] gmu [ dot ] edu)

## License
This repository is licensed under the MIT License. See `LICENSE` for details.

# Installing

```
git clone https://github.com/lucywilkerson/SWERVE
git clone --depth 1 https://github.com/lucywilkerson/SWERVE-data
git clone --depth 1 https://github.com/lucywilkerson/SWERVE-common
cd SWERVE
pip install -e .
```

If above install does not work due to error `proj executable not found.`, try

```
conda config --prepend channels conda-forge
conda config --set channel_priority strict
conda create -n python3.12-swerve pyproj python=3.12
conda activate python3.12-swerve
pip install -e .
```

# Using

See stdout for files read and written.

```
python info.py --config example.yaml # defines storm start/stop and creates info.csv, info.extended.csv, and info.extended.json with all site information
python main.py --config example.yaml # reads data for all sites, applies filters, generates all figures for all sites, and updates info.extended.csv/.json, see instructions below to generate for specific sites
```

To generate plots for just one site (e. g., Bull Run):
```
python main.py --config example.yaml --sites 'Bull Run'
# Can also use 'test' to create figures for test data:
# python main.py --config test.yaml --sites test
```

Config file (.yaml) in the configs subdirectory is used to define:
- run config name (names subdir for info files and log files, default is same as config file name)
- event (e.g. ['2015-03-17','2024-05-10']; if 'all', runs all events for given data source/type/class; if not specified, must have storm start/stop specified)
- start/stop time (optional, must be provided together)
- data source (e.g. ['TVA','INTERMAG','Watari'])
- data type (e.g. ['GIC','B','DMM']; if not defined, runs all)
- data class (e.g. ['measured','calculated']; if not defined, runs all)
- main.py keyword arguments (exclude errors, summary table, etc.)
- filter keyword arguments (for GIC measured data filtering)

# HAPI

The HAPI interface reads cached data from `SWERVE-data/data_processed` and
provides dataset IDs, metadata, and time series.

```python
from swerve.hapi.catalog import catalog, catalog_all
from swerve.hapi.info import info
from swerve.hapi.swerve_to_numpy import swerve_to_numpy

ids = catalog()
entries = catalog_all()
metadata = info(ids[0])
rows = swerve_to_numpy(
    'NERC', '10052', 'GIC', 'measured', True,
    '2024-05-10T15:00:00Z', '2024-05-10T15:03:00Z',
)
```

Dataset IDs have the form
`data_source/site_id/data_type/data_class/{original,modified}`. Each ID covers
the available event or annual partitions for that series. Only variants with
cached records are listed. IDs are ordered by measured/calculated class,
GIC/B/DMM type, provider, and site.

`catalog()` returns the ID list and writes `catalog.json` in the SWERVE project
root. `catalog_all()` returns `[{"id": ..., "info": ...}, ...]` and writes
`catalog-all.json` there. Pass `output=None` to either function to return the
result without writing, or `output=...` to choose another file.

`info(id)` returns time coverage, parameter labels, units, and source metadata.
The `additionalMetadata` field includes `source`, `siteID`, cache partition
details, and citations from `data_original/<source>/info.json`.

`swerve_to_numpy(...)` returns a NumPy object array with shape `(n, 2)`, where
`n` is the number of matching records. Column 0 contains UTC datetime objects.
Column 1 contains a numeric scalar for single-component data or a NumPy array
of values for multi-component data. If no records match, it returns an empty
array with shape `(0, 2)`.

The interval includes start and excludes stop. Naive timestamps are interpreted
as UTC. Missing values remain
NaN; missing files and years are skipped. Requests for modified data use only
that cache branch. Identical overlapping records are deduplicated; conflicting
values raise `ValueError`. Values are returned without resampling or filtering.

Annual sources read original records from `data_processed/non-event/<year>`
and modified records from storm caches. Other sources use event caches.
All four functions accept `data_root=...` to select a SWERVE-data checkout;
the default is the sibling directory. The checkout must contain processed
caches before these functions are called.

For deployment, generate `catalog-all.json` when caches or citation metadata
change, then load it once at startup:

```python
import json

with open('catalog-all.json') as handle:
    entries = json.load(handle)
metadata_by_id = {entry['id']: entry['info'] for entry in entries}
```

Request handlers can use `metadata_by_id[dataset_id]` for metadata and
`swerve_to_numpy(...)` for time series. The HTTP frontend supplies the HAPI
response wrapper and data serialization.

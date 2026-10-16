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

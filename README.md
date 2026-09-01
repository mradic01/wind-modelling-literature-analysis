# Meteorological Station Coverage Analysis

This branch contains the code and generated outputs used for the meteorological
station coverage analysis presented in Chapter 3 of the qualification exam thesis
_Artificial Intelligence and Machine Learning for Wind Modelling and Prediction:
Data Sources, Methods, and Research Gaps_.

The analysis compares the density and spatial spacing of active publicly indexed
meteorological stations across selected countries.

## Data sources

Station metadata are obtained from the Meteostat public station directory:

https://bulk.meteostat.net/v2/stations/lite.json.gz

Country land areas are obtained from the World Bank indicator
`AG.LND.TOTL.K2` (Land area, km²).

## Method

For each country, the script calculates:

- number of active Meteostat stations;
- number of stations containing a WMO identifier;
- station density per 1,000 km²;
- equivalent grid spacing;
- mean distance to the nearest station within the same country.

Nearest-station distances are calculated from station coordinates using
the Haversine metric.

The Meteostat station count represents stations available in the Meteostat
directory and should not be interpreted as a complete inventory of stations
operated by individual national meteorological services.

## Files

- `station_density.py` — analysis script.
- `output/station_density.csv` — tabulated results.
- `output/station_density.xlsx` — Excel version of the results.
- `output/station_density_table.tex` — LaTeX table used in the thesis.
- `output/station_density.png` — generated station-density figure.
- `output/station_density_metadata.txt` — methodological notes and data-source details.

## Requirements

```bash
pip install matplotlib numpy pandas scikit-learn openpyxl
```

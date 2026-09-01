# European Coastline Analysis

This branch contains the source code and generated outputs used for the
coastline analysis presented in Section 2.1 of the qualification exam thesis
_Artificial Intelligence and Machine Learning for Wind Modelling and Prediction:
Data Sources, Methods, and Research Gaps_.

The analysis compares mainland and island marine coastline lengths for selected
European countries using harmonised Eurostat GISCO 2024 spatial data.

## Methodology

The analysis uses Eurostat GISCO Countries 2024 spatial data at a scale of
1:1,000,000 (EPSG:4326).

For each country:

- only European territories are retained;
- the largest connected polygon is classified as the mainland;
- remaining marine polygons are classified as islands;
- official GISCO coastline segments are spatially matched to the corresponding
  polygons using a 250 m tolerance;
- coastline lengths are calculated geodesically using the WGS84 ellipsoid.

The resulting values are reproducible calculations based on harmonised GISCO
geometry and should not be interpreted as official national coastline totals.
Results depend on the spatial resolution and scale of the source dataset.

## Files

- `coastline_analysis.py` — complete Python workflow used for the analysis.
- `output/european_coastline_comparison.csv` — generated results in CSV format.
- `output/european_coastline_comparison.xlsx` — generated results in Excel format.
- `output/european_coastline_comparison.tex` — generated LaTeX table used in the thesis.

## Requirements

The analysis requires Python 3 and the following packages:

```bash
pip install geopandas shapely pyproj pandas openpyxl requests
```

# Wind Modelling Literature Analysis

Source code and supporting material for the analyses presented in the qualification exam thesis.

# AI Method Classification and Evolution

This branch contains the scripts and selected outputs used for the analysis of
Artificial Intelligence and Machine Learning methods presented in Chapter 6 of
the qualification exam thesis _Artificial Intelligence and Machine Learning for
Wind Modelling and Prediction: Data Sources, Methods, and Research Gaps_.

The analysis is divided into two parts:

- classification of AI and ML methods reported in the literature;
- analysis of the temporal evolution of individual AI methodologies.

## AI method classification

The `ai_method_classification` directory contains the script and input data used
to classify publications according to the AI and ML methods identified in the
bibliographic records.

The resulting classifications and aggregated method counts are provided in:

`output/ai_method_classification/`

## Evolution of AI methodologies

The `ai_method_evolution` directory contains the script and classified input
data used to analyse the temporal evolution of individual AI methods.

The resulting heatmap is provided in:

`output/ai_method_evolution/`

## Method

The analysis is based on rule-based keyword matching applied to bibliographic
records previously classified in the literature-analysis workflow.

The classification identifies individual methods and groups them into broader
AI categories, including classical machine learning, artificial neural networks,
deep learning, and physics-informed or hybrid AI approaches.

## Outputs

Selected outputs used in the thesis are included in the `output` directory,
including aggregated method counts and the temporal heatmap of AI methodologies.

## Requirements

The scripts require Python 3 and standard data-analysis and plotting packages:

```bash
pip install pandas openpyxl matplotlib
```

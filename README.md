# E-SARRIS: Ensemble SAR River Ice Surveillance

This repository contains the classification script and pipeline for generating probabilistic ice type maps using Sentinel-1 SAR imagery and Random Forest models. The system supports modular training, scene-specific classification, and metadata logging.

## Key Features

- Scene-specific training or inference with Sentinel-1 data
- Includes polarization ratio/difference and optional FDD
- Supports multi-class classification: Open Water, Smooth Ice, Ice, Rough Ice
- Logs metadata to XML (scene ID, date, AFDD, region, wind, class proportions)

## Usage

1. Set `sample_new_points = True` if using new training points.
2. Use `retrain_model = True` to merge scenes for retraining.
3. Adjust `sampling_layers = ['FDD']` as needed.
4. Run the script to classify and generate outputs:
    - Classified GeoTIFFs
    - Probability maps
    - XML logs with classification metadata

## Dependencies

See `requirements.txt` for Python package requirements.


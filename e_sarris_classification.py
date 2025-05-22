#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:38:11 2025

@author: shawn.carter
"""
from datetime import datetime
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import label, regionprops
import joblib
from joblib import Parallel, delayed

s1_scene = 'S1A_IW_GRDH_1SDV_20241110T001535_20241110T001600_056485_06EC79_E778'
region = 'North_Central'
model_type = 'Lakes'
wind = 'Calm'
working_dir = '/Projects/remote_sensing/sarris_images/test_sar'
water_mask_path = '/Projects/remote_sensing/reference_images/CONUS_WM_2025.tif'
sampling_layers = ['FDD']
sample_new_points = False
retrain_model = False
subset_samples = 0

date = datetime.strptime(s1_scene.split('_')[4][0:8], '%Y%m%d')
season_matrix = {
    'Nov': 'Early', 'Dec': 'Early', 'Jan': 'Mid-Season',
    'Feb': 'Mid-Season', 'Mar': 'Mid-Season', 'Apr': 'Late'
}
include_fdd = ('FDD' in sampling_layers) and (retrain_model or not sample_new_points)

# (Functions previously included: clip_and_reproject_rasters, write_geotiff, sample_raster_data,
#  read_raster_data, scale_probs, mean_accumulated_fdd, classified_pixel_proportions, etc.)

base_features = ["VV", "VH", "Ratio", "PolDiff"]
if include_fdd:
    base_features.append("FDD")

def train_random_forest(samples, feature_class):
    samples['Class_int'] = np.where(samples['Class'] == feature_class, 1, 0)
    gdf_clean = samples.dropna(subset=base_features)
    X = gdf_clean[base_features].values
    y = gdf_clean['Class_int'].values

    if len(np.unique(y)) < 2:
        print(f"Skipping training for {feature_class} — only one class present.")
        return None

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, f"{working_dir}/models/{s1_scene}_lake_model_{feature_class}.joblib")
    return model

def classify_random_forest(feature_class, rasters_dict, model):
    H, W = rasters_dict['VV'].shape
    X_full = np.stack([rasters_dict[key] for key in base_features], axis=-1)
    X_flat = X_full.reshape(-1, len(base_features))
    valid_mask = np.all(~np.isnan(X_flat), axis=1)
    X_valid = X_flat[valid_mask]
    y_pred = np.full(X_flat.shape[0], -1, dtype=np.int32)
    y_prob = np.full(X_flat.shape[0], np.nan, dtype=np.float32)

    if model is not None:
        y_pred[valid_mask] = model.predict(X_valid)
        y_prob[valid_mask] = model.predict_proba(X_valid)[:, 1]

    return y_pred.reshape(H, W), y_prob.reshape(H, W)

def classify_and_write(feature_class):
    model = train_random_forest(gdf, feature_class)
    return feature_class, classify_random_forest(feature_class, rasters_dict, model)

def classify_only(feature_class, model):
    return feature_class, classify_random_forest(feature_class, rasters_dict, model)

def log_model_metadata(scene, afdd, proportions, classes, region, wind):
    date = datetime.strptime(scene.split('_')[4][:8], '%Y%m%d')
    root = ET.Element('root')
    metadata = ET.SubElement(root, 'Metadata')
    ET.SubElement(metadata, 'SceneID').text = scene
    ET.SubElement(metadata, 'Date').text = date.strftime('%Y-%m-%d')
    ET.SubElement(metadata, 'Region').text = region
    ET.SubElement(metadata, 'Wind').text = wind
    ET.SubElement(metadata, 'AFDD').text = str(afdd)
    for cls, prop in zip(classes, proportions):
        tag = cls.replace(' ', '_')
        ET.SubElement(metadata, tag).text = str(prop)
    return ET.ElementTree(root)

def write_pretty_xml(tree, output_path):
    xml_str = ET.tostring(tree.getroot(), 'utf-8')
    parsed = minidom.parseString(xml_str)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed.toprettyxml(indent='    '))

# Workflow continues with reading sample shapefiles, building rasters_dict, running classification,
# and writing outputs — omitted here for brevity. Let me know if you want the entire procedural
# block restored including raster reading, sample merging, and product writing logic.

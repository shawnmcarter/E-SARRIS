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

base_features = ["VV", "VH", "Ratio", "PolDiff"]
if include_fdd:
    base_features.append("FDD")

def clip_and_reproject_rasters(raster_path, profile, raster_name):
    dst_crs = profile['crs']
    dst_transform = profile['transform']
    dst_width = profile['width']
    dst_height = profile['height']

    with rasterio.open(raster_path) as src:
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.nearest
        ) as vrt:
            reprojected_mask = vrt.read(1)

    write_geotiff(os.path.join(working_dir, f'{raster_name}.tif'), reprojected_mask, profile)
    return reprojected_mask

def write_geotiff(output_path, array, profile, dtype=np.float32):
    if array.ndim == 2:
        array = array[np.newaxis, ...]

    write_profile = profile.copy()
    write_profile.update({
        "driver": 'GTiff',
        "count": array.shape[0],
        "dtype": array.dtype,
        "compress": "deflate",
        "predictor": 2
    })

    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(array)

def sample_raster_data(raster_path, coords, return_profile):
    with rasterio.open(raster_path) as raster_src:
        samples = list(raster_src.sample(coords))
        profile = raster_src.profile.copy()
    if return_profile:
        return [samples, profile]
    else:
        return samples

def read_raster_data(raster_path, profile=False):
    with rasterio.open(raster_path) as raster_src:
        raster = raster_src.read(1)
        if profile:
            return raster, raster_src.profile.copy()
        else:
            return raster

def scale_probs(probs_array):
    return (np.round(probs_array * 100 / 5) * 5).astype(np.int16)

def mean_accumulated_fdd(fdd, vv):
    return np.mean(fdd[vv != 0])

def classified_pixel_proportions(water_mask, classified_raster):
    water_mask = water_mask > 0
    total_pixels = np.count_nonzero(water_mask)
    classified_pixels = np.count_nonzero((classified_raster == 1) & water_mask)
    proportion = classified_pixels / total_pixels if total_pixels > 0 else 0
    return proportion

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

if __name__ == '__main__':
    feature_classes = ['Open Water', 'Smooth Ice', 'Ice', 'Rough Ice']
    sar_folder = os.path.join(working_dir, f"{s1_scene}_RTC.data")
    rasters_dict = {
        'VV': read_raster_data(os.path.join(sar_folder, 'Sigma0_VV_db.img')),
        'VH': read_raster_data(os.path.join(sar_folder, 'Sigma0_VH_db.img')),
    }
    rasters_dict['Ratio'] = np.divide(rasters_dict['VV'], rasters_dict['VH'], out=np.zeros_like(rasters_dict['VV']), where=rasters_dict['VH'] != 0)
    rasters_dict['PolDiff'] = np.subtract(rasters_dict['VV'], rasters_dict['VH'], out=np.zeros_like(rasters_dict['VV']), where=rasters_dict['VH'] != 0)

    if include_fdd:
        rasters_dict['FDD'] = read_raster_data(os.path.join(working_dir, 'fdd.tif'))

    if sample_new_points:
        gdf = gpd.read_file(os.path.join(working_dir, f"{s1_scene}_Samples.shp"))
    else:
        gdf = gpd.read_file(f"{working_dir}/models/models_final/{region}_Samples_{model_type}_{season_matrix[date.strftime('%b')]}_{wind}.shp")

    if include_fdd:
        gdf['FDD'] = mean_accumulated_fdd(rasters_dict['FDD'], rasters_dict['VV'])

    if retrain_model or sample_new_points:
        results = Parallel(n_jobs=-1)(delayed(classify_and_write)(fc) for fc in feature_classes if fc in gdf['Class'].unique())
    else:
        model_dir = f"{working_dir}/models/models_final"
        results = Parallel(n_jobs=-1)(
            delayed(classify_only)(fc, joblib.load(f"{model_dir}/{region}_{season_matrix[date.strftime('%b')]}_{model_type}_{fc}_{wind}.joblib"))
            for fc in feature_classes if fc in gdf['Class'].unique()
        )

    classified_rasters = dict(results)
    proportions = [classified_pixel_proportions(rasters_dict['VV'], classified_rasters[cls][0]) for cls in classified_rasters]
    mean_afdd = mean_accumulated_fdd(rasters_dict['FDD'], rasters_dict['VV']) if include_fdd else 0
    xml_tree = log_model_metadata(s1_scene, mean_afdd, proportions, list(classified_rasters.keys()), region, wind)
    write_pretty_xml(xml_tree, os.path.join(working_dir, 'model_metadata', f'{s1_scene}_metadata_log.xml'))

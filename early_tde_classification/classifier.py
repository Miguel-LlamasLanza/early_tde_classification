#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:56:46 2025

@author: lmiguel
"""

import pandas as pd
import joblib
import os
import logging
import datetime as dt
from early_tde_classification.config import Config


def load_all_features():
	"""
	Load all feature data into a pd.DataFrame

	"""

	# Load feat data (including features, labels, and other info)
	feat_data = pd.read_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features.csv'),
							dtype = {"alertId": str})
	return feat_data


def filter_features(feat_data, feat_cols = ['amplitude', 'rise_time', 'temperature', 'r_chisq',
										   'sigmoid_dist', 'snr_rise_time', 'snr_amplitude'],
					metadata_cols = ['objectId', 'alertId', 'type']):
	"""
	Keep only the features that are used in the classfier.

	Parameters
	----------
	feat_data : pd.DataFrame
		All feature data including metadata.
	feat_cols : list, optional
		Features used by the model. The default is ['amplitude', 'rise_time', 'temperature',
									'r_chisq', 'sigmoid_dist', 'snr_rise_time', 'snr_amplitude'].
	metadata_cols : list, optional
		Metadata columns. The default is ['objectId', 'alertId', 'type'].

	Returns
	-------
	feat_data : pd.DataFrame
		Feature data including the metadata (object ID, type, etc).
	features : pd.DataFrame
		Features to be used by the classifier.

	"""
	# Check consistency
	if not all(col in feat_data.columns for col in feat_cols + metadata_cols):
		logging.warning('WARNING: Some feature is missing in the input data!!\nMissing feature(s):')
		logging.info([col for col in feat_cols + metadata_cols if col not in feat_data.columns])

	feat_data = feat_data[feat_cols + metadata_cols]

	# Get features without metadata
	features = feat_data[feat_cols]

	return feat_data, features


def load_features_for_classifier():
	"""
	Load features for classifier.

	Returns
	-------
	feat_data : pd.DataFrame
		Feature data including the metadata (object ID, type, etc).
	features : pd.DataFrame
		Features to be used by the classifier.

	"""
	feat_data = load_all_features()
	feat_data, features = filter_features(feat_data)

	return feat_data, features


def run_classifier_on_features(features, features_with_metadata, save_candidates = True):
	"""
	Classify based on the already extracted features.

	Parameters
	----------
	features : pd.DataFrame
		Features to be used by the classifier.
	features_with_metadata : pd.DataFrame
		Feature data including the metadata (object ID, type, etc).
	save_candidates : float, optional
		Whether to save the candidates in a csv file. The default is True.

	Returns
	-------
	candidate_tdes : list
		TDE candidates returned by the classifier.

	"""

	# Load model
	loaded_rf = joblib.load(os.path.join(Config.DATA_DIR, 'models', 'RF_TDE_classifier_depth_8.joblib'))

	# Predict label
	features_with_metadata['predicted_label'] = loaded_rf.predict(features)

	# Get TDE candidates
	candidate_tdes = features_with_metadata[features_with_metadata['predicted_label'] == 'TDE']
	if save_candidates:
		candidate_tdes.to_csv(os.path.join(Config.OUTPUT_DIR, 'candidate_tdes.csv'))

	return candidate_tdes.objectId.to_list()


if __name__ == '__main__':

	start = dt.datetime.now()

	features_with_metadata, features = load_features()
	candidate_tdes = run_classifier_on_features(features, features_with_metadata)
	logging .info("Done in {} seconds.".format(dt.datetime.now() - start))







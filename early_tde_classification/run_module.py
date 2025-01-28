#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:53:51 2025

@author: lmiguel
"""
import logging
import datetime as dt


try:
	from early_tde_classification.extract_features import load_data_and_extract_features
	from early_tde_classification.classifier import filter_features, run_classifier_on_features

except ModuleNotFoundError:  # This is in case the module is not installed.
	from extract_features import load_data_and_extract_features
	from classifier import filter_features, run_classifier_on_features


# Measure time
start = dt.datetime.now()

feature_data_all = load_data_and_extract_features(save = True, show_plots = False)
features_with_metadata, features = filter_features(feature_data_all)
candidate_tdes = run_classifier_on_features(features, features_with_metadata)

# Log the computation duration
logging .info("Done in {} seconds.".format(dt.datetime.now() - start))

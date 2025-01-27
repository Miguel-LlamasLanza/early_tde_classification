#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:54:05 2023

@author: lmiguel
"""
from os.path import join, dirname, abspath


class Config:
	"""
	Configuration file
	"""

	# FINK API URL
	fink_api_url = 'https://fink-portal.org'

	# Paths
	PROJECT_PATH = dirname(abspath(__file__))
	DATA_DIR = join(PROJECT_PATH, '..', 'data')
	INPUT_DIR = join(DATA_DIR, 'input')
# 	INPUT_DATA_FNAME = 'full_LCs_all_objects_tns_all.parquet'
	INPUT_DATA_FNAME = 'full_LCs_all_objects_tns_all.parquet'

	OUTPUT_DIR = join(DATA_DIR, 'output')
	OUT_FEATURES_DIR = join(OUTPUT_DIR, 'features')

	# Filters ZTF
	filt_conv = {1: "g", 2: "r", 3: "i"}  # Conversion between filter ID (int) and filter name (str)
	band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0}  # Bands in amstrong for rainbow

	# PARAMETERS
	days_history_lc = 100

	# Postfit cuts
	min_snr_features = 1.5
	max_rchisq = 10
	sigdist_lim = [0, 8]   # Sigmoid center distance limits.


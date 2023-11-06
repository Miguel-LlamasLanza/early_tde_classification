#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:54:05 2023

@author: lmiguel
"""
import os

class Config:
	"""
	Configuration file
	"""

	# FINK API URL
	fink_api_url = 'https://fink-portal.org'

	# Paths
	DATA_DIR = '/home/lmiguel/Data/TDE_classification/'
	ZENODO_DATA_DIR = os.path.join(DATA_DIR, 'AL_data/')
	OUT_FEATURES_DIR = 'Features_check'
	ZTF_TDE_DATA_DIR = 'ZTF_TDE_Data'

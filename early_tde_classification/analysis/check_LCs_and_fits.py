#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:50:30 2024

@author: Miguel

Script to check LCs and fits for a given set of objects
"""

from early_tde_classification import extract_features
import pandas as pd
import numpy as np

by_alert = True
if by_alert:
	# By alertID
	df_neighbours_alerts = pd.read_csv('neighbours_alerts.csv', usecols=["0", "1", "2", "3"])
	alert_list = np.ravel(df_neighbours_alerts.T).tolist()
	object_list = None
else:
	# By objectID
	df_neighbours_object = pd.read_csv('neighbours.csv', usecols=["0", "1", "2", "3"])
	object_list = np.ravel(df_neighbours_object.T).tolist()
	alert_list = None




extract_features.extract_features('all', max_nb_files_simbad = 2, keep_only_last_alert=True,
								  save = False, show_plots = True, object_list = object_list,
								  alert_list = alert_list)


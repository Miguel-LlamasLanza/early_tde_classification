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

df_neighbours = pd.read_csv('neighbours.csv')
# df_neighbours = pd.read_csv('neighbours_alerts.csv')

object_list = np.ravel(df_neighbours.T).tolist()
# object_list = ['ZTF20abfcszi']

extract_features.extract_features('all', max_nb_files_simbad = 2, keep_only_last_alert=True,
								  save = False, show_plots = True, object_list = object_list,
								  alert_list = None)


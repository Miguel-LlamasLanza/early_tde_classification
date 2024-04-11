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
import matplotlib.pyplot as plt

plt.figure()
plt.plot()

"""
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

"""


# object_list = ['ZTF20abeywdn', 'ZTF18aaohwsl', 'ZTF20abeezqx', 'ZTF18aatxncf',
#        'ZTF19aassnjd', 'ZTF20abdxoeu', 'ZTF20abfcpvs', 'ZTF18abacxpl',
#        'ZTF20abfhwlt', 'ZTF18aakzqjh']
# extract_features.extract_feature_extragalactic_obj_full_LC(show_plots = True, save = False, object_list = object_list)

df = pd.read_csv('/home/lmiguel/Data/TDE_classification/features/features_tdes_ztf.csv')
alert_list = df.alertId.tolist()
# alert_list = ['ZTF19acspeuw-7']

extract_features.extract_features('tdes_ztf', keep_only_last_alert=True,
								  save = False, show_plots = True, alert_list=alert_list)


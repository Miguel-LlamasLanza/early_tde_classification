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
#########################
"""
object_list = ['ZTF20abeywdn', 'ZTF18aaohwsl', 'ZTF20abeezqx', 'ZTF18aatxncf',
       'ZTF19aassnjd', 'ZTF20abdxoeu', 'ZTF20abfcpvs', 'ZTF18abacxpl',
       'ZTF20abfhwlt', 'ZTF18aakzqjh']
object_list = ['ZTF19abbxyxi']
object_list = ['ZTF18abclhwt']
object_list = ['ZTF18aahjvlj']
feat = extract_features.extract_feature_extragalactic_obj_full_LC(show_plots = True, save = False, object_list = object_list)
"""
#####################3
"""
df = pd.read_csv('/home/lmiguel/Data/TDE_classification/features/features_tdes_ztf.csv')
alert_list = df.alertId.tolist()
# alert_list = ['ZTF19acspeuw-7']

golden_sample = ['ZTF17aaazdba',
'ZTF18actaqdw',
'ZTF19aabbnzo',
'ZTF19aapreis',
'ZTF19aarioci',
'ZTF19abhhjcc',
'ZTF19abzrhgq',
'ZTF19accmaxo',
'ZTF20abfcszi',
'ZTF20abjwvae',
'ZTF20acitpfz',
'ZTF20acqoiyt']

object_list = [t for t in df.objId.tolist() if t not in golden_sample]
object_list = [t for t in df.objId.tolist() if t in golden_sample]

"""

##################3
alert_list = ['1308229225515015006',
 '1299339641815015024',
 '1313199070915015005',
 'ZTF19aarioci-11',
 '1280342250715010002',
 '1255201002415010002',
 'ZTF17aaazdba-20',
 '1284231002015015002',
 '1271211676115015017',
 '1292212741015015008',
 '1263225911315015000',
 '1290230744115015000',
 '1265264683815015002',
 '1333334823615015003']


# object_list = ['ZTF19abuwgfg',
#  'ZTF20acitpfz',
#  'ZTF20abejvpr',
#  'ZTF19aarioci',
#  'ZTF20abfehpe',
#  'ZTF20abmzfql',
#  'ZTF20abgfekk']

fname = '/home/lmiguel/gitlab_projects/TDE classification/tde_classification/\
early_tde_classification/neighbours/tde_candidates.csv'
df = pd.read_csv(fname)

object_list = df.objId.tolist()

object_list=['ZTF18acxfcda', 'ZTF17aaazdba']


# feat = extract_features.extract_features('tdes_ztf', keep_only_last_alert=True,
# 								  save = False, show_plots = True, object_list = object_list)



# feat = extract_features.extract_features('extragal', object_list = object_list,
# 								  save = False, show_plots = True)

feat = extract_features.extract_features('tdes_ztf', keep_only_last_alert=True,
								  save = False, show_plots = True)


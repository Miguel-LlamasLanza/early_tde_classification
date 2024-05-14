#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:35:31 2024

@author: Miguel Llamas Lanza
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from early_tde_classification.config import Config
import itertools
import seaborn as sns
# import cmasher as cmr

def get_data(path, object_list = None):

	data = pd.read_csv(path)
	data = data.replace(-999., -1)

	# Remove the duplicate TDE
	data = data[~((data.objId.isin(['ZTF20abfcszi'])) & (data.data_origin == 'tdes_ztf'))]

	if object_list:
		# Keep only the golden sample for TDE
		data = data[(data['objId'].isin(object_list)) | (data['type']!='TDE')]

	data.reset_index(inplace=True, drop=True)

	metadata = data['type'].to_numpy()

	to_drop = {'objId', 'alertId', 'type', 'data_origin','ref_time', 'err_ref_time',
			'err_rise_time', 'err_amplitude', 'err_temperature'}

	data.drop(columns=to_drop, inplace = True)
	return data, metadata

def scatter_and_hist(x_all, y_all, x_tde, y_tde, x_sn, y_sn):

	colors = ['#F5622E', '#15284F']
	plt.figure()
# 	plt.hexbin(x_all, y_all, mincnt=1, bins='log', gridsize=20, label = 'all')
	plt.scatter(x_tde, y_tde, marker='x', c=colors[0], label='TDE')
	plt.scatter(x_sn, y_sn, marker='+', c=colors[1], label='SN')
	plt.scatter(x_all, y_all, marker = '.', c=colors[1], label = 'all')

# 	plt.hist(feat_tdes[feature], histtype = 'step', lw = 3, label = 'TDEs', c=colors[0])
# 	plt.hist(feat_others[feature], histtype = 'step', lw = 3, label = 'non-TDEs')
# 	plt.hist(feat_sn[feature], histtype = 'step', lw = 3, label = 'SN')

	sns.jointplot(x=x_all, y=y_all)



	plt.legend()


golden = ['ZTF17aaazdba','ZTF19aabbnzo',
			  'ZTF19aapreis','ZTF19aarioci',
			  'ZTF19abhhjcc','ZTF19abzrhgq',
			  'ZTF20abfcszi','ZTF20abjwvae',
			  'ZTF20acitpfz','ZTF20acqoiyt']
path_features = os.path.join(Config.OUT_FEATURES_DIR, 'features_all.csv')



features, metadata = get_data(path_features, object_list = golden)
mask_tdes = metadata == 'TDE'
mask_others = metadata != 'TDE'
mask_sn = np.isin(metadata, ['Early SN Ia candidate', 'SN candidate'])
feat_tdes = features[metadata == 'TDE']
feat_others = features[metadata != 'TDE']

feat_sn = features[np.isin(metadata, ['Early SN Ia candidate', 'SN candidate'])]



features_in_log = ['rise_time', 'r_chisq', 'snr_rise_time', 'snr_amplitude', 'norm']


"""
for feature in features.columns:

	plt.figure()
	plt.title(feature)
	plt.hist(feat_tdes[feature], histtype = 'step', lw = 3, label = 'TDEs')
	plt.hist(feat_others[feature], histtype = 'step', lw = 3, label = 'non-TDEs')
	plt.hist(feat_sn[feature], histtype = 'step', lw = 3, label = 'SN')


	plt.yscale('log')

	if feature in features_in_log:
		plt.xscale('log')



	plt.xlabel(feature)
	plt.ylabel('#')
	plt.legend()
"""
# Density plots

for feat_1, feat_2 in itertools.product(features_in_log, features_in_log):
	print(feat_1, feat_2)

	if feat_1 != feat_2:

		scatter_and_hist(feat_tdes[feat_1], feat_tdes[feat_2], feat_others[feat_1], feat_others[feat_2],
				  feat_sn[feat_1], feat_sn[feat_2])
		plt.yscale('log')
		plt.xlabel(feat_1)
		plt.ylabel(feat_2)



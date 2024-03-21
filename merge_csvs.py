#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:02:34 2024

@author: Miguel

Merges all csvs into a single csv file.
"""

import glob
import pandas as pd
import os
import sys
from config import Config


all_csvs = glob.glob(os.path.join(Config.OUT_FEATURES_DIR, '*.csv'))
out_csv_fname = os.path.join(Config.OUT_FEATURES_DIR, 'features_all.csv')

if out_csv_fname in all_csvs:
	print('Results will overwrite the file (%s)'% out_csv_fname)
	user_input = input('continue? y/n  ')
	if user_input in ['y', 'Y', 'yes']:
		print('overwriting...')
		all_csvs.remove(out_csv_fname)
	else:
		sys.exit()


all_features = []
for csv_fname in all_csvs:
	df = pd.read_csv(csv_fname)
	if 'data_origin' not in df.columns:
		df['data_origin'] = os.path.basename(csv_fname)[9:-4]
	all_features.append(df)

# Concat
merged_df = pd.concat(all_features)
# Save
merged_df.to_csv(out_csv_fname, index = False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:20:40 2023

@author: lmiguel
"""

import pandas as pd
import os
import glob
import logging
import numpy as np
logging.basicConfig(encoding='utf-8', level=logging.INFO)



def load_data_other_objects():

	# TODO: Create config file and put paths there
	path_parquet_files = '/home/lmiguel/Data/TDE_classification/AL_data/'
	all_parquet_fnames = glob.glob(os.path.join(path_parquet_files, '*.parquet'))

	logging.info('Loading parquet files')
	all_obj_df = pd.concat([pd.read_parquet(parquet_fname) for parquet_fname in all_parquet_fnames])
	#all_obj_df.sort_values(['objectId', 'cjd'])

	# Keep only row with all alerts
	all_obj_df['length'] = all_obj_df['cjd'].apply(lambda x: len(x))
	all_obj_df.sort_values(['objectId', 'length'], inplace = True)
	names = np.array(all_obj_df['objectId'])
	mask = np.append((names[:-1] != names[1:]), True)
	all_obj_df = all_obj_df[mask]

	logging.info('All files loaded')

	return all_obj_df


all_obj_df = load_data_other_objects

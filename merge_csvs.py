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
from config import Config


all_csvs = glob.glob(os.path.join(Config.OUT_FEATURES_DIR, '*.csv'))

all_features = []
for csv_fname in all_csvs:
	all_features.append(pd.read_csv(csv_fname))

# Concat
merged_df = pd.concat(all_features)
# Save
merged_df.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features_all.csv'), index = False)

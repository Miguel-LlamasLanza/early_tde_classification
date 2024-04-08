#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:57:52 2024

@author: Miguel


Script to check some statistics about the SIMBAD objects that pass the cuts.
"""
import pandas as pd
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from early_tde_classification.config import Config



# Load data
# Data SIMBAD before cuts
nb_files = 2   # (Number of files extracted)
parquet_fnames = glob.glob(os.path.join(Config.ZENODO_DATA_DIR, '*simbad*.parquet'))[:nb_files]
df = pd.concat([pd.read_parquet(parquet_fname) for parquet_fname in parquet_fnames])
# df = df.drop_duplicates(subset='objectId', keep="first").copy()


# Data SIMBAD after cuts
features_simbad = pd.read_csv(os.path.join(Config.OUT_FEATURES_DIR, 'old', 'features_simbad.csv'))
# features_simbad_object = features_simbad.drop_duplicates(subset='objId', keep="first").copy()
features_simbad_object = features_simbad

"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [12, 8])

# Plot SIMBAD before cuts
df['cdsxmatch'].value_counts().head(30).plot(kind='bar', ax = ax1)
ax1.set_title('Before cuts')
ax1.set_yscale('log')


# Plot SIMBAD after cuts
ax2 = features_simbad_object['type'].value_counts().head(30).plot(kind='bar', ax = ax2, color = 'orange')
ax2.set_yscale('log')
ax2.set_ylabel("Number of objects")
ax2.set_title('After cuts')

features_simbad_object['type'].value_counts().head(30).index.to_list()

fig.subplots_adjust(hspace = 0.8)


"""

nb_categories = 40
plt.figure()
ax = features_simbad_object['type'].value_counts().head(nb_categories).plot(kind='bar',
												figsize = [10, 4], label = 'After cuts')
ax.set_yscale('log')
ax.set_ylabel("Number of objects")
ax.set_title('Categories SIMBAD data')

categories = features_simbad_object['type'].value_counts().head(nb_categories).index.to_list()

df['cdsxmatch'].value_counts()[categories].plot(kind='bar', ax = ax, color = 'grey', ls = '--',
												fill = False, label = 'Before cuts')

plt.legend()










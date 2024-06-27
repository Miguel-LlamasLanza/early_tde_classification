#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:33:30 2024

@author: Miguel

Plots lightcurves of all TDEs from FINK csv
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


from early_tde_classification.conversion_tools import mag2fluxcal_snana
from early_tde_classification.config import Config



fink_tdes = pd.read_csv(
	os.path.join(Config.PROJECT_PATH, '..', 'Archive or backup', 'ZTF_TDE_Data',
			  'from_Fink.csv'), dtype={'id': str})

fink_tdes.sort_values('jd', inplace = True)


for obj_id in np.unique(fink_tdes.objectId):

	obj_df = fink_tdes[fink_tdes.objectId == obj_id]

	# Convert filter list to g, r, i strings
	filt_list = np.vectorize(Config.filt_conv.get)(obj_df['fid'].astype(int))

	# Flux conversion
	flux, fluxerr = mag2fluxcal_snana(obj_df['magpsf'], obj_df['sigmapsf'])

	plt.figure(figsize=(12, 8))

	colors = ['green', 'red', 'black']
	for idx, flt_str in enumerate(['g', 'r', 'i']):
		flt_mask = filt_list == flt_str
		plt.errorbar(obj_df['jd'][flt_mask], flux[flt_mask], yerr=fluxerr[flt_mask], fmt='o',
			   alpha=.7, color=colors[idx], label = flt_str)
		plt.title(obj_id)
plt.show()

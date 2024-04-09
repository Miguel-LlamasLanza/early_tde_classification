#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:59:03 2024

@author: Miguel

Includes functions to plot anomaly scores (see also script snad_aad.py)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

from early_tde_classification.config import Config


def plot_hists(scores, mask_tdes, mask_tns, mask_simbad):

	fig, (ax1,ax2, ax3) = plt.subplots(1, 3, figsize = (15, 4))

	# Histograms
	ax1.hist(scores, bins=25, label = 'total', color = 'grey', histtype='step', lw = 1)
	# plt.hist(scores[mask_others], bins=25, color = '#15284F', label = 'non TDEs', alpha = 0.5)
	ax1.hist(scores[mask_simbad], bins=25, color = 'green', label = 'Simbad', alpha = 0.4)
	ax1.hist(scores[mask_tns], bins=25, color = 'blue', label = 'TNS', alpha = 1)
	ax1.hist(scores[mask_tdes], bins=25, color = 'red', label = 'TDEs', alpha = 1)
	# Set-up
	ax1.set_xlabel('anomaly scores', fontsize=14)
	ax1.set_ylabel('N', fontsize=14)
	ax1.legend()
	ax1.set_yscale('log')

	# --- Second plot ---

	# Histograms
	ax2.hist(scores, bins=25, label = 'total', color = 'grey', histtype='step', lw = 1)
	ax2.hist(scores[mask_simbad], bins=25, color = 'green', label = 'Simbad', alpha = 0.4)
	ax2.hist(scores[mask_tns], bins=25, color = 'blue', label = 'TNS')
	ax2.hist(scores[mask_tdes], bins=25, color = 'red', label = 'TDEs')
	# Set-up
	ax2.set_xlabel('anomaly scores', fontsize=14)
	ax2.legend()

	# --- Third plot ---

	# Histograms
	ax3.hist(scores, bins=25, label = 'total', color = 'grey', histtype='step', lw = 1)
	ax3.hist(scores[mask_simbad], bins=25, color = 'green', label = 'Simbad', alpha = 0.4)
	ax3.hist(scores[mask_tns], bins=25, color = 'blue', label = 'TNS')
	ax3.hist(scores[mask_tdes], bins=25, color = 'red', label = 'TDEs')
	# Set-up
	ax3.set_xlabel('anomaly scores', fontsize=14)
	ax3.set_ylim(top = 600)
	ax3.legend()


scores_pine = pd.read_csv(os.path.join(Config.ANOMALY_DET_DATA_DIR, 'scores_pine.csv'))
scores_iso = pd.read_csv(os.path.join(Config.ANOMALY_DET_DATA_DIR, 'scores_iso.csv'))

mask_tdes = scores_pine.labels == 'tdes_ztf'
mask_tns = scores_pine.labels == 'tns'
mask_simbad = scores_pine.labels == 'simbad'

plot_hists(scores_pine.scores, mask_tdes, mask_tns, mask_simbad)
plt.suptitle('Pineforest with AL')

mask_tdes = scores_iso.labels == 'tdes_ztf'
mask_tns = scores_iso.labels == 'tns'
mask_simbad = scores_iso.labels == 'simbad'

plot_hists(scores_iso.scores, mask_tdes, mask_tns, mask_simbad)
plt.suptitle('Isolation forest (no AL)')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:09:32 2024

@author: lmiguel
"""
import numpy as np
from early_tde_classification.config import Config
import matplotlib.pyplot as plt


def plot_lc_from_lc_values_array(lc_values):

	colors = ['#15284F', '#F5622E']
	filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))
	for idx, flt_str in enumerate(['g', 'r']):

		flt_mask = filt_list == flt_str

		plt.errorbar(lc_values[0][flt_mask], lc_values[1][flt_mask], yerr=lc_values[2][flt_mask], fmt='o', alpha=.7,
			   color=colors[idx], label = flt_str)
	plt.legend()
	plt.ylabel('Flux')
	plt.xlabel('Time (JD)')

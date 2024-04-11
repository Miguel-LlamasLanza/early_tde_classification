#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:20:22 2024

@author: Miguel
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('post_fit_temporal_features.csv', names = ['objectId', 'alertId', 'type',
															  'sigmoid_dist', 'snr_rise_time'])

df_tdes = df[df.type == 'TDE']
df_others = df[df.type != "TDE"]


plt.scatter(df_tdes.sigmoid_dist, df_tdes.snr_rise_time, label ='TDEs')
plt.scatter(df_others.sigmoid_dist, df_others.snr_rise_time, label = 'non TDEs')

plt.xlabel('sigmoid distance')
plt.ylabel('SNR rise time')

plt.yscale('log')
plt.xscale('log')

plt.legend()

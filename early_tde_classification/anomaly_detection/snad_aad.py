# -*- coding: utf-8 -*-
"""
Extracted from SNAD AAD notebook.
Full documentation of Corniferest can be found [ReadTheDocs](https://coniferest.readthedocs.io/en/latest/).



"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from early_tde_classification.config import Config

from coniferest.isoforest import IsolationForest
from coniferest.pineforest import PineForest
from coniferest.session import Session
from coniferest.session.callback import TerminateAfter, prompt_decision_callback


"""# Load Features (for the TDE example)

"""


def load_features(all_features_csv):

	features = pd.read_csv(all_features_csv)
	# Shuffle dataframe
	df = features.sample(frac = 1).reset_index(drop=True)
	# Get array with type of objects
	# metadata = df['type'].to_numpy()
	metadata = df['data_origin'].to_numpy()

	# Remove not scalar columns (and other extra columns)
	df = df.loc[:, (df.columns!='alertId') & (df.columns!='type') & (df.columns !='objId')
			 & (df.columns !='data_origin')]
	df = df.loc[:, (df.columns!='ref_time') & (df.columns!='err_ref_time')]

	return df, metadata


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


def get_percentages_above_treshold(scores, score_thresh):
	# Get percentages of each population above threshold (Make this as a funtion)
	simbad_above = np.sum(scores[mask_simbad] < score_thresh) / np.sum(scores < score_thresh)
	tns_above = np.sum(scores[mask_tns] < score_thresh) / np.sum(scores < score_thresh)
	tdes_above = np.sum(scores[mask_tdes] < score_thresh) / np.sum(scores < score_thresh)

	false_positive = np.sum(scores[mask_others] < score_thresh) / len(scores[mask_others])
	false_negative = np.sum(scores[mask_tdes] > score_thresh) / len(scores[mask_tdes])

	data = [[simbad_above, tns_above, tdes_above,
					false_positive, false_negative]]

	df = pd.DataFrame(data = data,
				   columns = ['Simbad above thresh', 'TNS above thresh', 'TDEs above thresh',
							'non TDEs above thresh', 'TDEs below thresh'])
	return df


# Load
df, labels = load_features(os.path.join(Config.OUT_FEATURES_DIR, 'features_all.csv'))
labels[labels == 'tdes_ztf'] = 'TDE'
feat_data_to_fit = df.to_numpy()   	# This is 2d np.array

# Create masks per type
mask_tdes = labels == 'TDE'
mask_others = labels != 'TDE'
mask_tns = labels == 'tns'
mask_simbad = labels == 'simbad'



""" Isolation forest (non active AD)

Let’s run Isolation Forest model (see [Liu *et al.* 2008](https://doi.org/10.1109/ICDM.2008.17)) on the data we just created:
"""

# declare and fit an isolation forest
model_iso = IsolationForest(random_seed=42)
model_iso.fit(feat_data_to_fit)

# evaluate classification
scores_iso = model_iso.score_samples(feat_data_to_fit)

# Save scores to csv file
df_scores_iso = pd.DataFrame(np.column_stack([scores_iso, labels]), columns = ['scores', 'labels'])
df_scores_iso.to_csv(os.path.join(Config.ANOMALY_DET_DATA_DIR, 'scores_iso.csv'), index = False)


"""
### Analysis of anomaly scores

"""
# Dsitribution
plot_hists(scores_iso, mask_tdes, mask_tns, mask_simbad)

# Analyse scores
# Get total percentage population
simbad_total = len(scores_iso[mask_simbad]) / len(scores_iso)
tns_total = len(scores_iso[mask_tns]) / len(scores_iso)
tdes_total = len(scores_iso[mask_tdes]) / len(scores_iso)

data_total = [simbad_total, tns_total, tdes_total]

df_iso = get_percentages_above_treshold(scores_iso, -0.5)


"""
## Test with Active anomaly detection (tune the AD algorithm to your definition of anomaly)"""


# declare the model parameters
model_ex1 = PineForest(
		# Use 1024 trees, a trade-off between speed and accuracy
		n_trees=1024,
		# Fix random seed for reproducibility
		random_seed=42,
)

# labels_integer = (labels == 'TDE').astype(int)
labels_integer = labels == 'TDE'

session_ex1 = Session(
		data=feat_data_to_fit,
		metadata=labels_integer,
		model=model_ex1,
		decision_callback=lambda metadata, data, session: metadata,
		on_decision_callbacks=TerminateAfter(3000),
)

session_ex1.run()

scores_pine = session_ex1.model.score_samples(feat_data_to_fit)
# Save
df_scores_pine = pd.DataFrame(np.column_stack([scores_pine, labels]), columns = ['scores', 'labels'])
df_scores_pine.to_csv('scores_pine.csv', index = False)

# Analysis
plot_hists(scores_pine, mask_tdes, mask_tns, mask_simbad)

df_pine = get_percentages_above_treshold(scores_pine, -0.5)
df_both = pd.concat([df_iso, df_pine])


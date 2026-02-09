import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
# from tqdm.auto import tqdm
import pandas as pd
from early_tde_classification.scripts_paper_version.functions import (
	get_tns_info, get_lc, request_lc_snad, plot_lc, deredden, apply_quality_cuts_to_features)
import os
from astropy.time import Time
import xgboost as xgb

# Definitions
COLORS_ZTF = {1: "#15284F", 2: "#F5622E", 3: "#3C8DFF", 4: "#D5D5D3"}

# Which objects to plot
example_objs = [
	'ZTF23abjvojy', # QSO, two very strong variability episodes almost fifteen years apart. Repeated Ambiguous Nuclear Transient?..
	'ZTF22abzajwl', # AT2023adr - repeating TDE
	'ZTF20accxwrk', # AT2020ukj - long-duration brightening of a until-then passive galaxy. Could be a TDE, but the timescales are very long.
	'ZTF19aayijkh', # AT2022yhf - long-duration brightening of a blue, slightly variable object that is likely an AGN. No cooling down in the decay. Hints of infrared echo.
]
# Paths
input_dir = '/home/lmiguel/gitlab_projects/TDE classification/tde_classification/data/input'
feat_dir ='/home/lmiguel/gitlab_projects/TDE classification/tde_classification/data/output/features'


# Load pre-computed features
features = Table.read(os.path.join(feat_dir, 'features_sergey_2025',
								   'features.all.pmulti.distnr.ext.vot'))

# Rename the temperature
features.rename_column('T', 'temperature')
features.rename_column('e_T', 'e_temperature')
features.rename_column('snr_T', 'snr_temperature')
# Remove column
features.remove_column('raw')

# features = apply_quality_cuts_to_features(features)
features = features.to_pandas()


# Load data
data = pd.read_parquet(os.path.join(input_dir, 'tde_objects_all_withupperlim_min5_1.parquet'),
					   filters=[('objectId', 'in', example_objs)])
data = data[data['objectId'].isin(example_objs)]


nearests = pd.read_parquet(os.path.join(input_dir, 'nearests_all_5.parquet'))
nearests.loc[nearests['distnr_g']<0, 'distnr_g'] = np.nan
nearests.loc[nearests['distnr_r']<0, 'distnr_r'] = np.nan

# Load classifiers
classifiers = []

for _ in ["model_nuclear.ubj", "model_broad.ubj"]:
	clf = xgb.XGBClassifier()
	clf.load_model(os.path.join(input_dir, "models", _))
	classifiers.append(clf)



plt.rcParams["font.size"] = 12

for i, cand in enumerate([{'objectId': _} for _ in example_objs]):
	print(cand['objectId'])

	full = get_lc(data[data['objectId'] == cand['objectId']].iloc[0])
	full = full[full['i:sigmapsf'] < 0.3]

	ra, dec = np.mean(data[data['objectId'] == cand['objectId']][['i:ra', 'i:dec']], axis=0)

	snad = request_lc_snad(ra, dec, 1.5)

	ax1 = plt.figure(figsize=(14, 5)).gca()
	plot_lc(full, show_limits=False, show_zero=True, ms=10)
	plt.grid(None)

	# SNAD
	if snad is not None:
		for c, cn, fid in ((COLORS_ZTF[1], 'g', 1), (COLORS_ZTF[2], 'r', 2)):
			idx = nearests['objectId'] == cand['objectId']
			idx &= (np.isfinite(nearests['magnr_' + cn])) & (nearests['distnr_' + cn] < 1.5)
			if np.sum(idx):
				flux0 = 10**(11 - 0.4 * np.mean(nearests['magnr_' + cn][idx]))
				flux0 = deredden(flux0, fid, ra, dec)
			else:
				flux0 = 0

			idx = snad['i:fid'] == fid
			plt.plot(snad['i:jd'][idx] - 2400000.5, snad['FLUXCAL'][idx] - flux0, '.',
					 ms=5, alpha=0.2, color=c)

	tns_info = get_tns_info(cand['objectId']) or 'no TNS info'
	plt.title(cand['objectId'] + '  :  ' + tns_info, fontsize = 19)
	plt.margins(0.02, 0.05)

#	 plt.ylim(None, 1.2*np.max(full['FLUXCAL']))

	plt.legend(loc='upper left')

	ax1.set_xlabel('Modified Julian Date', fontsize = 14)
	plt.ylabel('FLUXCAL', fontsize = 14)

	# Make a secondary x-axis with calendar year
	def mjd_to_year(x):
		return Time(x, format='mjd').decimalyear

	def year_to_mjd(x):
		return Time(x, format='decimalyear').mjd

	ax2 = ax1.secondary_xaxis("top", functions=(mjd_to_year, year_to_mjd))
	# ax2.set_xlabel("Year", fontsize = 14)

	# Check when is the object first identified with current model
	# Get features for this object
	feat_cand = features[features['objectId']==cand['objectId']]
	valid_score_mjd = []
	# Loop through features at each point (that passed previous filters)
	for i, row_feat in feat_cand.iterrows():
		for clf in classifiers:
			# Select features that are used in classifier
			feat_model = row_feat[clf.feature_names_in_]

			# Convert columns to numeric
			feat_model = pd.to_numeric(feat_model, errors='coerce')

			# Convert to a single-row DataFrame
			feat_model = pd.DataFrame([feat_model])

			# Run the classifier
			scores = clf.predict(feat_model)  # Best fit + sampled
			best_score = scores[0]  # Best fit score
			frac_scores = np.sum(scores) / len(scores)  # Fraction of samples that score positive

			if not best_score and frac_scores < 0.1:
				continue
			valid_score_mjd.append(row_feat['jd_max'] - 2400000.5)
	if len(valid_score_mjd)>0:
# 		for line in valid_score_mjd:
		ax1.axvline(np.min(valid_score_mjd), ls = 'dashed', color = 'black', lw = 1.2,
			  label = 'First identification')






	plt.savefig(f"fig_lc_{cand['objectId']}.pdf", bbox_inches='tight')
plt.show()

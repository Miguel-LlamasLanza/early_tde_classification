import requests
import numpy as np
# import sys
# sys.path.append('/home/lmiguel/gitlab_projects/tde_classification/spark/spark-3.1.3-bin-hadoop3.2/python')

import pandas as pd
from io import BytesIO
import tools_from_sn_classifier as sn_tools
from Data_other_studies.fink_sn_activelearning.actsnfink import classifier_sigmoid
# from fink_utils.spark.utils import concat_col

"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Spark intialization
spark = SparkSession.builder.appName("tde_classif").getOrCreate()
"""

def get_data_from_FINK(save = True):

	# Get object names
	TDEs_hammerstein = pd.read_fwf('ZTF_TDE_Data/Table1_Hammerstein', skiprows = 34, header = None)
	ztf_names = TDEs_hammerstein[1].to_list()

	what_to_retrieve = ['i:jd', 'i:fid', 'i:magpsf', 'i:sigmapsf', 'i:candid', 'i:objectId']

	r = requests.post('https://fink-portal.org/api/v1/objects',
	  json={
		  'objectId': ','.join(ztf_names),
		  'columns': ','.join(what_to_retrieve)
	  }
	)
	df = pd.read_json(BytesIO(r.content))

# 	df = df.rename(columns = {'i:jd': 'jd', 'i:fid': 'fid', 'i:magpsf': 'magpsf',
# 						   'i:sigmapsf': 'sigmapsf','i:candid': 'candid',
# 						   'i:objectId': 'objectId'})
	df.columns = df.columns.str.strip('i:')  # strip suffix at the right end only.

	df.to_csv('ZTF_TDE_Data/from_Fink.csv', index = None)
	return df


# get_data_from_FINK()

df = pd.read_csv('ZTF_TDE_Data/from_Fink.csv')
#df = spark.read.csv('ZTF_TDE_Data/from_Fink.csv', header = True)


"""
what_to_concat = ['jd', 'fid', 'magpsf', 'sigmapsf']

# Use for creating temp name
prefix = 'c'
# what_prefix = [prefix + i for i in what_to_concat]

# Append temp columns with historical + current measurements
for colname in what_to_concat:
 	df = concat_col(df, colname, prefix=prefix)
"""#



# TODO: Crop the light-curve before fitting



converted_df = sn_tools.convert_full_dataset(df, obj_id_header='objectId')


df_list = []
for indx in range(np.unique(converted_df['id'].values).shape[0]):

	name = np.unique(converted_df['id'].values)[indx]
	obj_flag = converted_df['id'].values == name

	for filt in ['g', 'r']:
		object_df = converted_df[obj_flag][converted_df['FLT'] == filt].copy()
		if len(object_df) > 2:
			tmax = object_df['MJD'][object_df['FLUXCAL'].idxmax()]

			object_df = object_df[object_df.MJD <= tmax]

			df_list.append(object_df)

converted_df_early = pd.concat(df_list)

feature_matrix = sn_tools.featurize_full_dataset(converted_df_early, screen = True)


feature_matrix.to_csv('Features_check/features_tdes.csv', index = None)
























"""    INFO ABOUT FINK FIELDS

# Columns starting with v: (ZTF original fields)
i_col = {'aimage': {'type': ['float', 'null'], 'doc': 'Windowed profile RMS afloat major axis from SExtractor [pixels]'},
 'aimagerat': {'type': ['float', 'null'], 'doc': 'Ratio: aimage / fwhm'},
 'bimage': {'type': ['float', 'null'], 'doc': 'Windowed profile RMS afloat minor axis from SExtractor [pixels]'},
 'bimagerat': {'type': ['float', 'null'], 'doc': 'Ratio: bimage / fwhm'},
 'candid': {'type': 'long', 'doc': 'Candidate ID from operations DB'},
 'chinr': {'type': ['float', 'null'], 'doc': 'DAOPhot chi parameter of nearest source in reference image PSF-catalog'},
 'chipsf': {'type': ['float', 'null'], 'doc': 'Reduced chi-square for PSF-fit'},
 'classtar': {'type': ['float', 'null'], 'doc': 'Star/Galaxy classification score from SExtractor'},
 'clrcoeff': {'type': ['float', 'null'], 'doc': 'Color coefficient from linear fit from photometric calibration of science image'},
 'clrcounc': {'type': ['float', 'null'], 'doc': 'Color coefficient uncertainty from linear fit (corresponding to clrcoeff)'},
 'clrmed': {'type': ['float', 'null'], 'doc': 'Median color of all PS1 photometric calibrators used from science image processing [mag]: for filter (fid) = 1, 2, 3, PS1 color used = g-r, g-r, r-i respectively'},
 'clrrms': {'type': ['float', 'null'], 'doc': 'RMS color (deviation from average) of all PS1 photometric calibrators used from science image processing [mag]'},
 'dec': {'type': 'double', 'doc': 'Declination of candidate; J2000 [deg]'},
 'decnr': {'type': 'double', 'doc': 'Declination of nearest source in reference image PSF-catalog; J2000 [deg]'},
 'diffmaglim': {'type': ['float', 'null'], 'doc': 'Expected 5-sigma mag limit in difference image based on global noise estimate [mag]'},
 'distnr': {'type': ['float', 'null'], 'doc': 'distance to nearest source in reference image PSF-catalog [pixels]'},
 'distpsnr1': {'type': ['float', 'null'], 'doc': 'Distance to closest source from PS1 catalog; if exists within 30 arcsec [arcsec]'},
 'distpsnr2': {'type': ['float', 'null'], 'doc': 'Distance to second closest source from PS1 catalog; if exists within 30 arcsec [arcsec]'},
 'distpsnr3': {'type': ['float', 'null'], 'doc': 'Distance to third closest source from PS1 catalog; if exists within 30 arcsec [arcsec]'},
 'drb': {'type': ['float', 'null'], 'doc': 'RealBogus quality score from Deep-Learning-based classifier; range is 0 to 1 where closer to 1 is more reliable'},
 'drbversion': {'type': 'string', 'doc': 'version of Deep-Learning-based classifier model used to assign RealBogus (drb) quality score'},
 'dsdiff': {'type': ['float', 'null'], 'doc': 'Difference of statistics: dsnrms - ssnrms'},
 'dsnrms': {'type': ['float', 'null'], 'doc': 'Ratio: D/stddev(D) on event position where D = difference image'},
 'elong': {'type': ['float', 'null'], 'doc': 'Ratio: aimage / bimage'},
 'exptime': {'type': ['float', 'null'], 'doc': 'Integration time of camera exposure [sec]'},
 'fid': {'type': 'int', 'doc': 'Filter ID (1=g; 2=R; 3=i)'},
 'field': {'type': ['int', 'null'], 'doc': 'ZTF field ID'},
 'fwhm': {'type': ['float', 'null'], 'doc': 'Full Width Half Max assuming a Gaussian core, from SExtractor [pixels]'},
 'isdiffpos': {'type': 'string', 'doc': 't or 1 => candidate is from positive (sci minus ref) subtraction; f or 0 => candidate is from negative (ref minus sci) subtraction'},
 'jd': {'type': 'double', 'doc': 'Observation Julian date at start of exposure [days]'},
 'jdendhist': {'type': ['double', 'null'], 'doc': 'Latest Julian date of epoch corresponding to ndethist [days]'},
 'jdendref': {'type': 'double', 'doc': 'Observation Julian date of latest exposure used to generate reference image [days]'},
 'jdstarthist': {'type': ['double', 'null'], 'doc': 'Earliest Julian date of epoch corresponding to ndethist [days]'},
 'jdstartref': {'type': 'double', 'doc': 'Observation Julian date of earliest exposure used to generate reference image [days]'},
 'magap': {'type': ['float', 'null'], 'doc': 'Aperture mag using 14 pixel diameter aperture [mag]'},
 'magapbig': {'type': ['float', 'null'], 'doc': 'Aperture mag using 18 pixel diameter aperture [mag]'},
 'magdiff': {'type': ['float', 'null'], 'doc': 'Difference: magap - magpsf [mag]'},
 'magfromlim': {'type': ['float', 'null'], 'doc': 'Difference: diffmaglim - magap [mag]'},
 'maggaia': {'type': ['float', 'null'], 'doc': 'Gaia (G-band) magnitude of closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [mag]'},
 'maggaiabright': {'type': ['float', 'null'], 'doc': 'Gaia (G-band) magnitude of closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [mag]'},
 'magnr': {'type': ['float', 'null'], 'doc': 'magnitude of nearest source in reference image PSF-catalog [mag]'},
 'magpsf': {'type': 'float', 'doc': 'Magnitude from PSF-fit photometry [mag]'},
 'magzpsci': {'type': ['float', 'null'], 'doc': 'Magnitude zero point for photometry estimates [mag]'},
 'magzpscirms': {'type': ['float', 'null'], 'doc': 'RMS (deviation from average) in all differences between instrumental photometry and matched photometric calibrators from science image processing [mag]'},
 'magzpsciunc': {'type': ['float', 'null'], 'doc': 'Magnitude zero point uncertainty (in magzpsci) [mag]'},
 'mindtoedge': {'type': ['float', 'null'], 'doc': 'Distance to nearest edge in image [pixels]'},
 'nbad': {'type': ['int', 'null'], 'doc': 'number of prior-tagged bad pixels in a 5 x 5 pixel stamp'},
 'ncovhist': {'type': 'int', 'doc': 'Number of times input candidate position fell on any field and readout-channel going back to beginning of survey'},
 'ndethist': {'type': 'int', 'doc': 'Number of spatially-coincident detections falling within 1.5 arcsec going back to beginning of survey; only detections that fell on the same field and readout-channel ID where the input candidate was observed are counted. All raw detections down to a photometric S/N of ~ 3 are included.'},
 'neargaia': {'type': ['float', 'null'], 'doc': 'Distance to closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [arcsec]'},
 'neargaiabright': {'type': ['float', 'null'], 'doc': 'Distance to closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [arcsec]'},
 'nframesref': {'type': 'int', 'doc': 'Number of frames (epochal images) used to generate reference image'},
 'nid': {'type': ['int', 'null'], 'doc': 'Night ID'},
 'nmatches': {'type': 'int', 'doc': 'Number of PS1 photometric calibrators used to calibrate science image from science image processing'},
 'nmtchps': {'type': 'int', 'doc': 'Number of source matches from PS1 catalog falling within 30 arcsec'},
 'nneg': {'type': ['int', 'null'], 'doc': 'number of negative pixels in a 5 x 5 pixel stamp'},
 'objectId': {'type': 'string', 'doc': 'object identifier or name'},
 'objectidps1': {'type': ['long', 'null'], 'doc': 'Object ID of closest source from PS1 catalog; if exists within 30 arcsec'},
 'objectidps2': {'type': ['long', 'null'], 'doc': 'Object ID of second closest source from PS1 catalog; if exists within 30 arcsec'},
 'objectidps3': {'type': ['long', 'null'], 'doc': 'Object ID of third closest source from PS1 catalog; if exists within 30 arcsec'},
 'pdiffimfilename': {'type': ['string', 'null'], 'doc': 'filename of positive (sci minus ref) difference image'},
 'pid': {'type': 'long', 'doc': 'Processing ID for science image to facilitate archive retrieval'},
 'programid': {'type': 'int', 'doc': 'Program ID: encodes either public, collab, or caltech mode'},
 'programpi': {'type': ['string', 'null'], 'doc': 'Principal investigator attached to program ID'},
 'publisher': {'type': 'string', 'doc': 'origin of alert packet'},
 'ra': {'type': 'double', 'doc': 'Right Ascension of candidate; J2000 [deg]'},
 'ranr': {'type': 'double', 'doc': 'Right Ascension of nearest source in reference image PSF-catalog; J2000 [deg]'},
 'rb': {'type': ['float', 'null'], 'doc': 'RealBogus quality score from Random Forest classifier; range is 0 to 1 where closer to 1 is more reliable'},
 'rbversion': {'type': 'string', 'doc': 'version of Random Forest classifier model used to assign RealBogus (rb) quality score'},
 'rcid': {'type': ['int', 'null'], 'doc': 'Readout channel ID [00 .. 63]'},
 'rfid': {'type': 'long', 'doc': 'Processing ID for reference image to facilitate archive retrieval'},
 'schemavsn': {'type': 'string', 'doc': 'schema version used'},
 'scorr': {'type': ['double', 'null'], 'doc': 'Peak-pixel signal-to-noise ratio in point source matched-filtered detection image'},
 'seeratio': {'type': ['float', 'null'], 'doc': 'Ratio: difffwhm / fwhm'},
 'sgmag1': {'type': ['float', 'null'], 'doc': 'g-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'sgmag2': {'type': ['float', 'null'], 'doc': 'g-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'sgmag3': {'type': ['float', 'null'], 'doc': 'g-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'sgscore1': {'type': ['float', 'null'], 'doc': 'Star/Galaxy score of closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star'},
 'sgscore2': {'type': ['float', 'null'], 'doc': 'Star/Galaxy score of second closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star'},
 'sgscore3': {'type': ['float', 'null'], 'doc': 'Star/Galaxy score of third closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star'}, 'sharpnr': {'type': ['float', 'null'], 'doc': 'DAOPhot sharp parameter of nearest source in reference image PSF-catalog'}, 'sigmagap': {'type': ['float', 'null'], 'doc': '1-sigma uncertainty in magap [mag]'},
 'sigmagapbig': {'type': ['float', 'null'], 'doc': '1-sigma uncertainty in magapbig [mag]'},
 'sigmagnr': {'type': ['float', 'null'], 'doc': '1-sigma uncertainty in magnr [mag]'},
 'sigmapsf': {'type': 'float', 'doc': '1-sigma uncertainty in magpsf [mag]'},
 'simag1': {'type': ['float', 'null'], 'doc': 'i-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'simag2': {'type': ['float', 'null'], 'doc': 'i-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'simag3': {'type': ['float', 'null'], 'doc': 'i-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'sky': {'type': ['float', 'null'], 'doc': 'Local sky background estimate [DN]'},
 'srmag1': {'type': ['float', 'null'], 'doc': 'r-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'srmag2': {'type': ['float', 'null'], 'doc': 'r-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'srmag3': {'type': ['float', 'null'], 'doc': 'r-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'ssdistnr': {'type': ['float', 'null'], 'doc': 'distance to nearest known solar system object if exists within 30 arcsec [arcsec]'},
 'ssmagnr': {'type': ['float', 'null'], 'doc': 'magnitude of nearest known solar system object if exists within 30 arcsec (usually V-band from MPC archive) [mag]'},
 'ssnamenr': {'type': ['string', 'null'], 'doc': 'name of nearest known solar system object if exists within 30 arcsec (from MPC archive)'},
 'ssnrms': {'type': ['float', 'null'], 'doc': 'Ratio: S/stddev(S) on event position where S = image of convolution: D (x) PSF(D)'},
 'sumrat': {'type': ['float', 'null'], 'doc': 'Ratio: sum(pixels) / sum(|pixels|) in a 5 x 5 pixel stamp where stamp is first median-filtered to mitigate outliers'},
 'szmag1': {'type': ['float', 'null'], 'doc': 'z-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]'},
 'szmag2': {'type': ['float', 'null'], 'doc': 'z-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]'}, 'szmag3': {'type': ['float', 'null'], 'doc': 'z-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]'}, 'tblid': {'type': ['long', 'null'], 'doc': 'Internal pipeline table extraction ID'}, 'tooflag': {'type': ['int', 'null'], 'doc': '1 => candidate is from a Target-of-Opportunity (ToO) exposure; 0 => candidate is from a non-ToO exposure'}, 'xpos': {'type': ['float', 'null'], 'doc': 'x-image position of candidate [pixels]'}, 'ypos': {'type': ['float', 'null'], 'doc': 'y-image position of candidate [pixels]'},
 'zpclrcov': {'type': ['float', 'null'], 'doc': 'Covariance in magzpsci and clrcoeff from science image processing [mag^2]'},
 'zpmed': {'type': ['float', 'null'], 'doc': 'Magnitude zero point from median of all differences between instrumental photometry and matched photometric calibrators from science image processing [mag]'}}


# Fields starting with v (FINK derived)
v_cols = {'classification': {'type': 'string', 'doc': 'Fink inferred classification. See https://fink-portal.org/api/v1/classes'},
'constellation': {'type': 'string', 'doc': 'Name of the constellation an alert on the sky is in'},
'firstdate': {'type': 'string', 'doc': 'Human readable datetime for the first detection of the object (from the i:jdstarthist field).'},
'g-r': {'type': 'double', 'doc': 'Last g-r measurement for this object.'},
'lapse': {'type': 'string', 'doc': 'Number of days between first and last detection.'},
'lastdate': {'type': 'string', 'doc': 'Human readable datetime for the alert (from the i:jd field).'},
'rate(g-r)': {'type': 'double', 'doc': 'g-r rate in mag/day (between last and first available g-r measurements).'}}

# Fields starting with d (Fink science module outputs)
d_cols = {'DR3Name': {'type': 'string', 'doc': 'Unique source designation of closest source from Gaia catalog; if exists within 1 arcsec.'},
 'Plx': {'type': 'double', 'doc': 'Absolute stellar parallax (in milli-arcsecond) of the closest source from Gaia catalog; if exists within 1 arcsec.'},
'cdsxmatch': {'type': 'string', 'doc': 'Object type of the closest source from SIMBAD database; if exists within 1 arcsec. See https://fink-portal.org/api/v1/classes'},
'e_Plx': {'type': 'double', 'doc': 'Standard error of the stellar parallax (in milli-arcsecond) of the closest source from Gaia catalog; if exists within 1 arcsec.'},
'gcvs': {'type': 'string', 'doc': 'Object type of the closest source from GCVS catalog; if exists within 1 arcsec.'},
'mulens': {'type': 'double', 'doc': 'Probability score of an alert to be a microlensing event by [LIA](https://github.com/dgodinez77/LIA).'},
'rf_kn_vs_nonkn': {'type': 'double', 'doc': 'Probability of an alert to be a Kilonova using a PCA & Random Forest Classifier (binary classification). Higher is better.'},
'rf_snia_vs_nonia': {'type': 'double', 'doc': 'Probability of an alert to be a SNe Ia using a Random Forest Classifier (binary classification). Higher is better.'},
'roid': {'type': 'int', 'doc': 'Determine if the alert is a potential Solar System object (experimental). 0: likely not SSO, 1: first appearance but likely not SSO, 2: candidate SSO, 3: found in MPC.'},
'snn_sn_vs_all': {'type': 'double', 'doc': 'The probability of an alert to be a SNe vs. anything else (variable stars and other categories in the training) using SuperNNova'},
'snn_snia_vs_nonia': {'type': 'double', 'doc': 'The probability of an alert to be a SN Ia vs. core-collapse SNe using SuperNNova'},
'vsx': {'type': 'string', 'doc': 'Object type of the closest source from VSX catalog; if exists within 1 arcsec.'}}

# Fields starting with b (ZTF cutouts)

b_cols = {'cutoutScience_stampData': {'type': 'array', 'doc': '2D array from the Science cutout FITS'},
'cutoutTemplate_stampData': {'type': 'array', 'doc': '2D array from the Template cutout FITS'},
'cutoutDifference_stampData': {'type': 'array', 'doc': '2D array from the Difference cutout FITS'}}
 """

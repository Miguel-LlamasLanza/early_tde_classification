import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


golden = ['ZTF17aaazdba','ZTF19aabbnzo',
			  'ZTF19aapreis','ZTF19aarioci',
			  'ZTF19abhhjcc','ZTF19abzrhgq',
			  'ZTF20abfcszi','ZTF20abjwvae',
			  'ZTF20acitpfz','ZTF20acqoiyt']


def get_data(path, only_golden_TDEs = True):

	data = pd.read_csv(path, dtype={'alertId': str})
	data = data.replace(-999., -1)


	if only_golden_TDEs:
		# Keep only the golden sample for TDE
		data = data[(data['objId'].isin(golden)) | (data['type']!='TDE')]


	data.reset_index(inplace=True, drop=True)

	to_drop = {'objId', 'alertId', 'type', 'data_origin','ref_time', 'err_ref_time',
	   'err_amplitude', 'err_rise_time', 'err_temperature', 'std_flux_g', 'std_flux_g'}

	features = data.copy().drop(columns=to_drop)

	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(features)
	scaled_features = np.array(scaled_features)

	return data, scaled_features

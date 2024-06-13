import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler


golden = ['ZTF17aaazdba','ZTF19aabbnzo',
              'ZTF19aapreis','ZTF19aarioci',
              'ZTF19abhhjcc','ZTF19abzrhgq',
              'ZTF20abfcszi','ZTF20abjwvae',
              'ZTF20acitpfz','ZTF20acqoiyt']


def get_data(path, cuts=True):

    data = pd.read_csv(path, dtype={'alertId': str})
    data = data.replace(-999., -1)

    # Remove the duplicate TDE
    data = data[~((data.objId.isin(['ZTF20abfcszi'])) & (data.data_origin == 'extragal'))]

    # Keep only the golden sample for TDE
    data = data[(data['objId'].isin(golden)) | (data['type']!='TDE')]

    if cuts:
        cuts = (data['temperature'] > 1e4) &\
        (data['rise_time'] < 1e2) &\
        (data['amplitude'] < 9.9999) &\
        (data['snr_rise_time'] > 1.5) &\
        (data['snr_amplitude'] > 1.5) &\
        (data['r_chisq'] < 1e2) &\
        (data['sigmoid_dist'] < 8) &\
        (data['sigmoid_dist'] > 0)

        data = data[cuts]

    data.reset_index(inplace=True, drop=True)

    to_drop = {'objId', 'alertId', 'type', 'data_origin','ref_time', 'err_ref_time',
       'err_amplitude', 'err_rise_time', 'err_temperature'}

    features = data.copy().drop(columns=to_drop)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = np.array(scaled_features)

    return data, scaled_features


def run_KDE(data, features, n=5):

    tdes = data[data['type'] == 'TDE'].index
    tree = KDTree(features)
    dist, ind = tree.query(features[tdes], k=n)
    dist = dist[:, 1:]
    ind = ind[:, 1:]

    objids = np.array([data.iloc[k]['objId'].values for k in [i for i in ind]])
    alertids = np.array([data.iloc[k]['alertId'].values for k in [i for i in ind]])
    types = np.array([data.iloc[k]['type'].values for k in [i for i in ind]])
    is_TDE = [np.in1d(TDE, tdes) for TDE in ind]

    return types, objids, alertids, is_TDE

def plot_distributions(data, types, n=10):

    new_dist = np.array(Counter(types.flatten()).most_common())
    distribution = np.array(Counter(data['type']).most_common())
    newtypes = np.in1d(distribution[:, 0], new_dist[:, 0])

    print('Neighbours of the TDEs:')
    print(new_dist)

    dist_plot_val = 100 * np.array(distribution[:, 1][newtypes], dtype='i') / len(data['type'])
    dist_plot_names = distribution[:, 0][newtypes]
    original = [dist_plot_val, dist_plot_names, "Original distribution"]

    new_plot_val = 100 * np.array(new_dist[:, 1], dtype='i') / len(types.flatten())
    new_plot_names = new_dist[:, 0]
    new = [new_plot_val, new_plot_names, "Nearest neighbours distribution"]

    for ds in [original, new]:
        plt.figure(figsize=(12, 8))
        plt.bar(ds[1][:n], ds[0][:n])
        plt.ylabel('Fraction of the set (%)')
        plt.title(ds[2])
        plt.xticks(rotation=30)

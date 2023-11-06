import pandas as pd
from actsnfink import *
from light_curve.light_curve_py import RainbowFit
import numpy as np
import glob

source = 'simbad'

if source == 'tns':

    # these files you can get from zenodo: https://zenodo.org/records/5645609#.YcD3przMJNg
    flist = glob.glob('/media/emille/git/Fink/supernova_al/data/AL_data/fink_cross_' + source + '_*.parquet')

    data_list = []
    for fname in flist:
        # read data from Fink
        data_temp = pd.read_parquet(fname)
        data_list.append(data_temp)

    data = pd.concat(data_list, ignore_index=True)


elif source == 'simbad':
     # these files you can get from zenodo: https://zenodo.org/records/5645609#.YcD3przMJNg
    ym = '202103'
    fname = '/media/emille/git/Fink/supernova_al/data/AL_data/fink_cross_' + source + '_' + ym +'.parquet'
    data = pd.read_parquet(fname)

# add TNS flag for SIMBAD objects
if 'TNS' not in data.keys():
    data['TNS'] = -99

# convert to flux
data_flux = convert_full_dataset(data, keep_objid=True)
data_flux.dropna(inplace=True)
data_flux.drop_duplicates(inplace=True, ignore_index=True)

# get rainbow features
features = fit_rainbow_dataset(data_flux, rising_criteria='diff', with_baseline=False, 
                                band_wave_aa={"g": 4770.0, "r": 6231.0}, min_data_points=5,
                                bar=False)

if source == 'simbad':
    # change location to outside github
    features.to_parquet('data/' + source + '_fixed_temp/rainbow_features_' + source + '_' + ym + '.parquet')
elif source == 'tns':
    features.to_parquet('data/' + source + '_fixed_temp/rainbow_features_' + source + '_201911_202103.parquet')



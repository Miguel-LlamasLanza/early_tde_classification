#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:04:49 2023

@author: lmiguel

This script contains old functions which are not used anymore.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import logging
from tqdm import tqdm



# First, several functions from the preliminary LC fitting methods (by Erwan)

def convert_mag_to_flux(tab_mag,tab_mag_err):
    """Used to convert from magnitude to flux,see Leoni+22 for details.
	Note: this conversion is different than the one that we are actually using now.
	"""
    tab_flux = 10^(-4*np.array(tab_mag)/10 +11)
    tab_flux_err = (4*np.log(10)/10)*np.array(tab_mag_err)*tab_flux
    return tab_flux,tab_flux_err

def sigmoid_profile(tab_time_since_zero, a, b, c):
    """Sigmoid shape of rising profile of TDEs. a corresponds to the slope, b to the time-offset, and c to the amplitude"""
    return c/(1+np.exp(-a*(tab_time_since_zero-b)))

def get_initial_values(tab_time_since_zero, tab_fluxes):
    """Used to guess initial values, to initialize the fit. The commented values are those from Leoni+22.
    They have been changed (using 25th and 90th percentile instead of max and min) to be more resilient to outliers"""
    c0 = np.nanmax(tab_fluxes)

    #a0=(np.nanmax(tab_fluxes) - np.nanmin(tab_fluxes))/(tab_time_since_zero[np.nanargmax(tab_fluxes)])
    #b0=(1/a0)*np.log((c0/np.nanmin(tab_fluxes))-1)
    a0 = (tab_fluxes[np.argsort(tab_fluxes)[9*len(tab_fluxes)//10]] - tab_fluxes[np.argsort(tab_fluxes)[len(tab_fluxes)//4]])/\
         (tab_time_since_zero[np.argsort(tab_fluxes)[9*len(tab_fluxes)//10]] - tab_time_since_zero[np.argsort(tab_fluxes)[len(tab_fluxes)//4]])#
    a0=a0/c0
    b0 = tab_time_since_zero[np.argsort(tab_fluxes)[len(tab_fluxes)//2]]#
    return a0,b0,c0

def noisy_sigmoid_profile(tab_time_since_zero, a, b, c, noise_level):
    """Used to generate synthetic noisy lightcurves, to test the fitting"""
    return sigmoid_profile(tab_time_since_zero, a, b, c)+np.random.normal(0,noise_level, len(tab_time_since_zero))

def test_fitting():
    """Generate random lightcurves, cut them so that we don't go too long after the peak, and then fit the sigmoid."""
    tab_times = np.linspace(0,20,10)
    true_b = np.random.uniform(-10,20)
    true_c = np.random.uniform(500,10000)
    true_a = np.random.uniform(1,5)
    peak_index = np.searchsorted(tab_times,true_b - (1/true_a)*np.log((1/0.99)-1))

    tab_fluxes = noisy_sigmoid_profile(tab_times, true_a, true_b, true_c, 50)
    tab_flux_err = np.full(len(tab_times), 50)
    if peak_index>0:
        tab_fluxes=tab_fluxes[:peak_index + 2]
        tab_flux_err=tab_flux_err[:peak_index + 2]
        tab_times = tab_times[:peak_index+2]


    if len(tab_fluxes)>3:
        tab_times = tab_times[tab_fluxes>0]
        tab_flux_err = tab_flux_err[tab_fluxes>0]
        tab_fluxes = tab_fluxes[tab_fluxes>0]

        a0, b0, c0 = get_initial_values(tab_times, tab_fluxes)
        #b0 = tab_times[len(tab_times)//2]
        plt.errorbar(tab_times, tab_fluxes, yerr=tab_flux_err, fmt='o')
        plt.plot(tab_times, sigmoid_profile(tab_times, a0, b0, c0), label='Initial Guess')

        popt, pcov = curve_fit(sigmoid_profile, xdata=tab_times, ydata=tab_fluxes, sigma=tab_flux_err,
                               p0=[a0, b0, c0])

        nbr_resample=1000
        sampling_precision =  3000
        new_pars = np.random.multivariate_normal(popt,pcov,nbr_resample)
        new_time = np.linspace(tab_times[0],tab_times[-1],sampling_precision)
        lightcurves = np.array([sigmoid_profile(new_time,*realization) for realization in new_pars])
        percentiles = np.nanpercentile(lightcurves, [16,50,84], axis=0)

        plt.plot(tab_times, sigmoid_profile(tab_times,*popt), label='Best Fit')
        plt.fill_between(new_time, percentiles[0], percentiles[2], alpha=0.2)
        #plt.plot(new_time, percentiles[1],alpha=0.5, label='Median fit')
        plt.legend()
        #plt.close()
    else:
        print("The peak was too close to the start !")

def load_ztf_lightcurves():
    """Load the ZTF lightcurves downloaded from Fig 17 of Hammerstein+22"""
    path_to_lc = "/ZTF_TDE_Data/"
    dics_data = []
    for lc in tqdm(os.listdir(path_to_lc)):
        #print(lc)
        if lc.endswith(".dat"):
            dics_data.append({"Name":lc.split(".")[0]})
            dics_data[-1]["Time_g"]=[]
            dics_data[-1]["Flux_g"]=[]
            dics_data[-1]["FluxErr_g"]=[]
            dics_data[-1]["Time_r"]=[]
            dics_data[-1]["Flux_r"]=[]
            dics_data[-1]["FluxErr_r"]=[]
            with open(os.path.join(path_to_lc,lc)) as file:
                lines = file.readlines()
                count =0
                for line in lines:
                    if count > 0:
                        splitted=line.split(" ")
                        splitted=[elt for elt in splitted if elt!='']
                        if splitted[1]=="g.ztf":
                            dics_data[-1]["Time_g"].append(float(splitted[0]))
                            dics_data[-1]["Flux_g"].append(float(splitted[2]))
                            dics_data[-1]["FluxErr_g"].append(float(splitted[3]))
                        elif splitted[1]=="r.ztf":
                            dics_data[-1]["Time_r"].append(float(splitted[0]))
                            dics_data[-1]["Flux_r"].append(float(splitted[2]))
                            dics_data[-1]["FluxErr_r"].append(float(splitted[3]))
                    count += 1
    return dics_data


def fitting_ztf_data():
    """Fit the ZTF lightcurves from Hammerstein+22. Bugs on some of the lightcurves."""
    dics_data=load_ztf_lightcurves()
    for dic in dics_data:
        plt.figure()
        for tab_time, tab_flux, tab_fluxerr, color in zip([dic['Time_g'],dic['Time_r']],
                                                                      [dic['Flux_g'],dic['Flux_r']],
                                                                      [dic['FluxErr_g'],dic['FluxErr_r']],
                                                                      ['b','r']):
            tab_flux=np.array(tab_flux)
            tab_time=np.array(tab_time)[tab_flux>0]
            tab_fluxerr=np.array(tab_fluxerr)[tab_flux>0]
            tab_flux=np.array(tab_flux)[tab_flux>0]

            peak_index = np.nanargmax(tab_flux)

            if peak_index>2:
                tab_time = tab_time[:peak_index+3]
                tab_flux = tab_flux[:peak_index+3]
                tab_fluxerr = tab_fluxerr[:peak_index+3]
                a0, b0, c0 = get_initial_values(tab_time, tab_flux)
                popt, pcov = curve_fit(sigmoid_profile, xdata=tab_time, ydata=tab_flux, sigma=tab_fluxerr,
                                       p0=[a0, b0, c0], maxfev=5000)

                nbr_resample = 1000
                sampling_precision = 3000
                new_pars = np.random.multivariate_normal(popt, pcov, nbr_resample)
                new_time = np.linspace(tab_time[0], tab_time[-1], sampling_precision)
                lightcurves = np.array([sigmoid_profile(new_time, *realization) for realization in new_pars])
                percentiles = np.nanpercentile(lightcurves, [16, 50, 84], axis=0)

                plt.errorbar(tab_time, tab_flux, yerr=tab_fluxerr, fmt='o', c=color, ecolor="gray", markeredgecolor="gray")
                plt.plot(new_time, sigmoid_profile(new_time, *popt), c=color)
                plt.fill_between(new_time, percentiles[0], percentiles[2], alpha=0.2, facecolor=color)


# Here, some functions that were used in the script generate_features_TDEs.py before.

def crop_lc_based_on_csv_values(df):
	"""
	Used to crop the lightcurves to retain only the rising part, based on values from csv (manual crop)

	Parameters
	----------
	df : pd.DataFrame
		Dataset with columns ['objectId', 'type', 'MJD', 'FLT','FLUXCAL', 'FLUXCALERR'].
		Each row is an alert.

	Returns
	-------
	converted_df_early : pd.DataFrame
		Cropped data, to retain only alerts in rising part of the LC.

	"""

	# Hard-code csv filename and read
	csv_fname = 'ZTF_TDE_Data/forced_photometry/TimeParametersTDEs_training.csv'
	times_csv = pd.read_csv(csv_fname)

	# Crossmatch csvs and cut LCs
	merged_csv = df.merge(times_csv, left_on = 'fp_fname', right_on = 'Filename')

	converted_df_early = df[(merged_csv.MJD >= merged_csv['Start (MJD)'] + 2400000.5)
						  & (merged_csv.MJD <= merged_csv['Peak (MJD)'] + 2400000.5)]
	return converted_df_early


def convert_diff_fp_to_magpsf(forcediffimflux, forcediffimfluxunc, zpdiff, SNT=3, SNU=5,
							  set_to_nan=True):
	"""
	Get magpsf and sigmapsf from forced photometry parameters. Function provided by Julien.
	"""
	if (forcediffimflux / forcediffimfluxunc) > SNT:
		# we have a confident detection, compute and plot mag with error bar:
		mag = zpdiff - 2.5 * np.log10(forcediffimflux)
		err = 1.0857 * forcediffimfluxunc / forcediffimflux
	else:
		# compute flux upper limit and plot as arrow:
		if not set_to_nan:
			mag = zpdiff - 2.5 * np.log10(SNU * forcediffimfluxunc)
		else:
			mag = np.nan
		err = np.nan

	return mag, err

def load_data_other_objects():
	"""
	Load data TNS and Simbad, into dataframe with only one row per alert. One value per alert.

	Returns
	-------
	all_obj_df : pd.DataFrame
		DataFrame with all alerts of all objects.

	"""

	# TODO: Create config file and put paths there
	path_parquet_files = '/home/lmiguel/Data/TDE_classification/AL_data/'
	all_parquet_fnames = glob.glob(os.path.join(path_parquet_files, '*tns*.parquet'))

	logging.info('Loading parquet files')
	all_obj_df = pd.concat([pd.read_parquet(parquet_fname) for parquet_fname in all_parquet_fnames])
	#all_obj_df.sort_values(['objectId', 'cjd'])

	# Remove duplicate values
	for col in ['cjd', 'cmagpsf', 'csigmapsf', 'cfid']:

		all_obj_df[col.lstrip('c')] = all_obj_df[col].apply(lambda row: row[-1])
		all_obj_df.drop(columns = col, inplace = True)

	all_obj_df = all_obj_df[all_obj_df.objectId.isin(all_obj_df.objectId.unique())].copy()
	logging.info('All files loaded')

	return all_obj_df

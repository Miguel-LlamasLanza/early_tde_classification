import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def convert_mag_to_flux(tab_mag,tab_mag_err):
    """Used to convert from magnitude to flux,see Leoni+22 for details"""
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


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from fit_Lightcurve import *

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



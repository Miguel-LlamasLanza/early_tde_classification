import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import csv as csv
import os
from astropy.io import fits, ascii
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm


def estimate_sn_tde_kdes(plot=False):
    """Returns the KDE for rise time & temperature of both TDEs and SNe. If plot==True, you get a plot out of it"""

    path = os.path.join(os.getcwd(), 'ZTF_TDE_Data/features_all.csv')

    #Read and filter the data
    data = ascii.read(path)
    data = data[(data['rise_time']<1e2)&(data['amplitude']<9.9999)\
                &(data['snr_rise_time']>1.5)&(data['snr_amplitude']>1.5)&(data['r_chisq']<1e2)\
                &(data['snr_rise_time']>1)&(data['sigmoid_dist']<8)&(data['sigmoid_dist']>0)]#(data['temperature']>1e4)&

    #Select the features you want to build KDE on
    features = ['rise_time', 'temperature']

    #Defines the two classes
    sn_data = data[(data['type']=='SN candidate')|(data['type']=='Early SN Ia candidate')]
    tde_data = data[data['type']=='TDE']

    #Builds the KDEs on the log of the parameters
    sn_kde = gaussian_kde(np.array([ np.log10(sn_data[features[0]]),np.log10(sn_data[features[1]])]))
    tde_kde = gaussian_kde(np.array([ np.log10(tde_data[features[0]]),np.log10(tde_data[features[1]])]))

    if plot:
        logTmin, logTmax = 3.5, 4.5
        log_rise_time_min, log_rise_time_max = 0, 1.5
        X, Y = np.mgrid[log_rise_time_min:log_rise_time_max:100j, logTmin:logTmax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        fig, axes = plt.subplots(1,3)
        axes[0].scatter( np.log10(sn_data[features[0]]), np.log10(sn_data[features[1]]), label='Supernovae', c='r', edgecolor='w')
        axes[0].scatter( np.log10(tde_data[features[0]]), np.log10(tde_data[features[1]]), label='TDEs', c='b', edgecolor='w')
        Z = np.reshape(sn_kde(positions).T, X.shape)
        axes[0].imshow(np.rot90(Z), cmap='cmr.ember',
                  extent=[log_rise_time_min, log_rise_time_max, logTmin, logTmax])
        axes[0].legend()
        axes[0].set_xlabel(r'log($\tau$)')
        axes[0].set_ylabel(r'log($T$)')
        axes[0].set_aspect("auto")
        axes[0].set_title('Supernovae KDE')
        # axes[0].loglog()

        axes[1].scatter( np.log10(sn_data[features[0]]), np.log10(sn_data[features[1]]), label='Supernovae', c='r', edgecolor='w')
        axes[1].scatter( np.log10(tde_data[features[0]]), np.log10(tde_data[features[1]]), label='TDEs', c='b', edgecolor='w')
        Z = np.reshape(tde_kde(positions).T, X.shape)
        axes[1].imshow(np.rot90(Z), cmap='cmr.ocean',
                  extent=[log_rise_time_min, log_rise_time_max, logTmin, logTmax])
        axes[1].legend()
        axes[1].set_xlabel(r'log($\tau$)')
        axes[1].set_ylabel(r'log($T$)')
        axes[1].set_aspect("auto")
        axes[1].set_title('TDEs KDE')


        axes[2].scatter(np.log10(sn_data[features[0]]), np.log10(sn_data[features[1]]), label='Supernovae', c='r', edgecolor='w')
        axes[2].scatter(np.log10(tde_data[features[0]]), np.log10(tde_data[features[1]]), label='TDEs', c='b', edgecolor='w')
        Z = np.reshape(tde_kde(positions).T/sn_kde(positions).T, X.shape)
        m=axes[2].imshow(np.rot90(Z), cmap='cmr.redshift_r',
                  extent=[log_rise_time_min, log_rise_time_max, logTmin, logTmax], norm=LogNorm(1e-1,1e1))
        c=plt.colorbar(ax=axes[2],mappable=m)
        c.set_label('Likelihood TDE / Likelihood SN')
        axes[2].legend()
        axes[2].set_xlabel(r'log($\tau$)')
        axes[2].set_ylabel(r'log($T$)')
        axes[2].set_aspect("auto")
        axes[2].set_title('Likelihood ratio map')

        # axes[1].loglog()

    return sn_kde, tde_kde

def likelihood_ratio(rise_time, temperature):
    """Returns the ratio of the KDEs for the given parameters. Used to classify between TDE and SN. The threshold is
    arbitrary - the larger the value (above 1), the more likely the object is a TDE rather than SN. Multiplying
    Precisely, multiplying by the prior ratio of TDE vs SN (so relative occurence rate in the data of both transients)
    gives you the Bayes factor, ie the ratio of probabilities of being TDE vs being SN."""
    sn_kde, tde_kde = estimate_sn_tde_kdes(plot=False)
    return tde_kde((np.log10(rise_time),np.log10(temperature)))/sn_kde((np.log10(rise_time),np.log10(temperature)))

rise_time, temperature = 5, 15000
print(likelihood_ratio(rise_time, temperature))
rise_time, temperature = 5, 5000
print(likelihood_ratio(rise_time, temperature))

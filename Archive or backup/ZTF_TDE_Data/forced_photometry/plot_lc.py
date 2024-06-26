""" Plot forced photometry lightcurve
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import argparse
import glob

def diff_phot(forcediffimflux, forcediffimfluxunc, zpdiff, SNT=3, SNU=5, set_to_nan=True):
	"""
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

def apparent_flux(magpsf, sigmapsf, magnr, sigmagnr, magzpsci):
	""" Compute apparent flux from difference magnitude supplied by ZTF
	This was heavily influenced by the computation provided by Lasair:
	https://github.com/lsst-uk/lasair/blob/master/src/alert_stream_ztf/common/mag.py
	Paramters
	---------
	fid
		filter, 1 for green and 2 for red
	magpsf,sigmapsf; floats
		magnitude from PSF-fit photometry, and 1-sigma error
	magnr,sigmagnr: floats
		magnitude of nearest source in reference image PSF-catalog
		within 30 arcsec and 1-sigma error
	magzpsci: float
		Magnitude zero point for photometry estimates
	isdiffpos: str
		t or 1 => candidate is from positive (sci minus ref) subtraction;
		f or 0 => candidate is from negative (ref minus sci) subtraction

	Returns
	--------
	dc_flux: float
		Apparent magnitude
	dc_sigflux: float
		Error on apparent magnitude
	"""
	if magpsf is None:
		return None, None

	# reference flux and its error
	magdiff = magzpsci - magnr
	if magdiff > 12.0:
		magdiff = 12.0
	ref_flux = 10**(0.4 * magdiff)
	ref_sigflux = (sigmagnr / 1.0857) * ref_flux

	magdiff = magzpsci - magpsf
	if magdiff > 12.0:
		magdiff = 12.0
	difference_flux = 10**(0.4 * magdiff)
	difference_sigflux = (sigmapsf / 1.0857) * difference_flux

	dc_flux = ref_flux + difference_flux

	# assumes errors are independent. Maybe too conservative.
	dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

	return dc_flux, dc_sigflux

def dc_mag(magpsf, sigmapsf, magnr, sigmagnr, magzpsci):
	""" Compute apparent magnitude from difference magnitude supplied by ZTF
	Parameters
	Stolen from Lasair.
	----------
	fid
		filter, 1 for green and 2 for red
	magpsf,sigmapsf
		magnitude from PSF-fit photometry, and 1-sigma error
	magnr,sigmagnr
		magnitude of nearest source in reference image PSF-catalog
		within 30 arcsec and 1-sigma error
	magzpsci
		Magnitude zero point for photometry estimates
	isdiffpos
		t or 1 => candidate is from positive (sci minus ref) subtraction;
		f or 0 => candidate is from negative (ref minus sci) subtraction
	"""
	dc_flux, dc_sigflux = apparent_flux(
		magpsf, sigmapsf, magnr, sigmagnr, magzpsci
	)

	# apparent mag and its error from fluxes
	if (dc_flux == dc_flux) and dc_flux > 0.0:
		dc_mag = magzpsci - 2.5 * np.log10(dc_flux)
		dc_sigmag = dc_sigflux / dc_flux * 1.0857
	else:
		dc_mag = np.nan
		dc_sigmag = np.nan

	return dc_mag, dc_sigmag


def plot_lc_for_file(fname, args):

	pdf = pd.read_csv(fname, comment='#', sep=' ')

	pdf = pdf\
		.drop(columns=['Unnamed: 0'])\
		.rename(lambda x: x.split(',')[0], axis='columns')


	if args.units == 'mag':
		magpsf, sigmapsf = np.transpose(
			[
				diff_phot(*args_, SNT=3, SNU=5, set_to_nan=args.quality_cuts) for args_ in zip(
					pdf['forcediffimflux'],
					pdf['forcediffimfluxunc'].values,
					pdf['zpdiff'].values,
				)
			]
		)

		mag_dc, err_dc = np.transpose(
			[
				dc_mag(*args_) for args_ in zip(
					magpsf,
					sigmapsf,
					pdf['nearestrefmag'].values,
					pdf['nearestrefmagunc'].values,
					pdf['zpmaginpsci'].values,
				)
			]
		)

		fig = plt.figure(figsize=(15, 7))
		for filt in np.unique(pdf['filter']):
			mask = pdf['filter'] == filt
			if args.quality_cuts:
				# Keep only measurements with flag = 0
				mask *= pdf['infobitssci'] == 0
			mask *= err_dc == err_dc
			sub = pdf[mask]
			plt.errorbar(
				sub['jd'].apply(lambda x: x - 2400000.5),
				mag_dc[mask],
				err_dc[mask],
				ls='',
				marker='o',
				label=filt
			)

		fig.gca().invert_yaxis()
		plt.legend();
		plt.title('DC magnitude from forced photometry')
		plt.xlabel('Modified Julian Date [UTC]')
		plt.ylabel('DC magnitude');

	elif args.units == 'flux':
		fig = plt.figure(figsize=(15, 7))
		for filt in np.unique(pdf['filter']):
			mask = pdf['filter'] == filt
			if args.quality_cuts:
				# Keep only measurements with flag = 0
				mask *= pdf['infobitssci'] == 0
			mask *= pdf['forcediffimfluxunc'].values > 0
			sub = pdf[mask]
			plt.errorbar(
				sub['jd'].apply(lambda x: x - 2400000.5),
				sub['forcediffimflux'],
				sub['forcediffimfluxunc'],
				ls='',
				marker='o',
				label=filt
			)

		plt.legend();
		plt.title('Difference flux from forced photometry')
		plt.xlabel('Modified Julian Date [UTC]')
		plt.ylabel('DC flux');

	else:
		print('Units not provided!!')




parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
	'-filename', type=str, default='',
	help="""
	Path to file containing forced photometry
	"""
)
parser.add_argument(
	'-units', type=str, default='flux',
	help="""
	Unit system: `mag` or `flux`
	"""
)
parser.add_argument(
	'--quality_cuts', action="store_true",
	help="""
	If specified, apply quality cuts
	"""
)
parser.add_argument(
	'--plot_all', action="store_true",
	help="""
	If specified, plot all files in this folder
	"""
)

args = parser.parse_args(None)
print(args)

if args.plot_all:
	files_fp = glob.glob('*.txt')
	if len(files_fp) > 10:
		usr_input = input('More than 40 figures will be created ({} files). Confirm (y)?'.format(len(files_fp)))
		if usr_input != 'y':
			sys.exit()

	for fname in files_fp:
		plot_lc_for_file(fname, args)
else:
	plot_lc_for_file(args.filename, args)

plt.show()


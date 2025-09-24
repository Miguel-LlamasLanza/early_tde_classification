import pandas as pd
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord, concatenate
import requests
import numpy as np
import matplotlib.pyplot as plt
import io

sfd = SFDQuery()
# Coefficients for ZTF filters for G23 extinction law from dust_extinction for Rv=3.1
Av = {1: 3.681, 2: 2.635, 3: 1.944}

COLORS_ZTF = {1: "#15284F", 2: "#F5622E", 3: "#3C8DFF", 4: "#D5D5D3"}


def deredden(flux, fid, ra, dec):
	# De-reddening using SFD from dustmaps
	Ebv = sfd(SkyCoord(ra, dec, unit='deg'))

	return flux / 10**(-0.4*(Ebv*Av[fid]))

def deredden_pdf(pdf, ra=None, dec=None):
	# De-reddening using SFD from dustmaps
	Ebv = sfd(SkyCoord(ra, dec, unit='deg'))
	for fid in [1,2,3]:
		idx = pdf['i:fid'] == fid
		pdf.loc[idx, 'FLUXCAL'] /= 10**(-0.4*(Ebv*Av[fid]))
		pdf.loc[idx, 'FLUXCALERR'] /= 10**(-0.4*(Ebv*Av[fid]))

		if 'FLUXCALUPPER' in pdf:
			pdf.loc[idx, 'FLUXCALUPPER'] /= 10**(-0.4*(Ebv*Av[fid]))


def get_lc(row):
	# Extract light curve from the data row
	pdf = pd.DataFrame({_: row[_] for _ in ['i:fid', 'i:jd', 'i:magpsf', 'i:sigmapsf', 'i:isdiffpos', 'FLUXCAL', 'FLUXCALERR', 'i:ra', 'i:dec']})

	if 'i:diffmaglim' in row:
		pdf['i:diffmaglim'] = row['i:diffmaglim']
		pdf['FLUXCALUPPER'] = 10**(11 - 0.4*pdf['i:diffmaglim'])

	pdf['i:mjd'] = pdf['i:jd'] - 2400000.5
	pdf.sort_values('i:jd', inplace=True, ignore_index=True)

	# Remove other bands?..
	pdf = pdf[(pdf['i:fid'] == 1) | (pdf['i:fid'] == 2)]

	deredden_pdf(pdf, row['i:ra'], row['i:dec'])
	pdf['i:ra'], pdf['i:dec'] = row['i:ra'], row['i:dec']

	return pdf

def plot_lc(pdf, show_zero=False, show_limits=False, ms=None, err_alpha=0.2, err_lw=None):
	for fid,cn,c in [(1, 'g', COLORS_ZTF[1]), (2, 'r', COLORS_ZTF[2])]:
		idx = pdf['i:fid'] == fid
		mjd = pdf['i:jd'] - 2400000.5

		plt.errorbar(mjd[idx], pdf['FLUXCAL'][idx], pdf['FLUXCALERR'][idx], fmt='.', c=c, alpha=err_alpha, lw=err_lw)
		plt.plot(mjd[idx], pdf['FLUXCAL'][idx], '.', color=c, alpha=1, ms=ms, label=cn)

		if show_limits and 'FLUXCALUPPER' in pdf:
			idx &= ~np.isfinite(pdf['FLUXCAL'])
			plt.plot(mjd[idx], pdf['FLUXCALUPPER'][idx], '^', alpha=0.3, color=c)

	plt.grid(alpha=0.2)
	if show_zero:
		plt.axhline(0, ls='--', color='black', alpha=0.2)


def mag2fluxcal(magpsf, sigmapsf, isdiffpos=None):
	if magpsf is None:
		return None, None

	fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
	fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

	if isdiffpos is not None:
		sign = np.where(isdiffpos == 'f', -1, 1)
	else:
		sign = 1

	return sign*fluxcal, fluxcal_err

def request_lc_snad(ra, dec, sr_arcsec=1.5):
	r = requests.get(
		"https://db.ztf.snad.space/api/v3/data/latest/circle/full/json",
		params={'ra': ra, 'dec': dec, 'radius_arcsec': sr_arcsec},
	)

	lc = []
	for k,v in r.json().items():
		lc1 = pd.DataFrame(v['lc'])
		lc1['filter'] = v['meta']['filter']
		lc1['i:fid'] = {'zg': 1, 'zr': 2, 'zi': 3}.get(v['meta']['filter'])
		lc.append(lc1)

	if len(lc):
		pdf = pd.concat(lc, ignore_index=True)

		pdf['i:jd'] = pdf['mjd'] + 2400000.5
		pdf['FLUXCAL'],pdf['FLUXCALERR'] = mag2fluxcal(pdf['mag'], pdf['magerr'])
		deredden_pdf(pdf, ra, dec)
	else:
		pdf = None

	return pdf

def get_tns_info(ztf_name):
	result = []

	r = requests.post(
			'https://api.fink-portal.org/api/v1/resolver',
			json={
				'resolver': 'tns',
				'reverse': True,
				'name': ztf_name
			}
			)

	# Format output in a DataFrame
	pdf = pd.read_json(io.BytesIO(r.content))

	for i,row in pdf.iterrows():
		res = row['d:fullname']
		if row['d:type'] and pd.notna(row['d:type']):
			res += ' (' + row['d:type']
			if row['d:redshift'] > 0:
				res += f" z={row['d:redshift']}"
			res += ')'

		if res not in result:
			result.append(res)

	return " | ".join(result)


def apply_quality_cuts_to_features(features):


	mask = (
		(features["r_chisq"] <= 10) &
		(features["e_reference_time"] <= 100) &
		(features["norm_rel_reference_time"] >= -10) &
		(features["norm_rel_reference_time"] <= 1) &
		(features["snr_rise_time"] >= 1.5) &
		(features["snr_temperature"] >= 1.5)
	)

	# Filter the dataframe
	features = features[mask]
	return features



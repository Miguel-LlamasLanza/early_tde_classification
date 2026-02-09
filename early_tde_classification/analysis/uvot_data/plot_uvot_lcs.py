import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd

# === Paths ===
input_root = "/home/lmiguel/Data/TDE_classification/uvot_data/output/"
df = pd.read_csv("/home/lmiguel/gitlab_projects/TDE classification/tde_classification/early_tde_classification/analysis/uvot_data/uvot_matches.csv")

plots_subdir = os.path.join(input_root, "plots")
os.makedirs(plots_subdir, exist_ok=True)

# === Constants ===
MJDREF = 51910.0  # Swift mission epoch (2001-01-01)
filter_colors = {
	"uvw2": "blue",
	"uvm2": "purple",
	"uvw1": "cyan",
	"u": "green",
	"b": "orange",
	"v": "red"
}

# === Loop over transients ===
transients = df["name"].unique()

for transient in transients:
	obsids = df.loc[df["name"] == transient, "obsid"].astype(str).str.zfill(11).tolist()
	plt.figure(figsize=(10, 5))
	print(f"📈 Plotting lightcurve for {transient} ...")

	for obsid in obsids:
		lc_files = glob.glob(os.path.join(input_root, obsid, f"{obsid}_*.lc"))
		for lc_file in lc_files:
			with fits.open(lc_file) as hdul:
				hdr = hdul[1].header
				data = hdul[1].data
				if not all(col in data.names for col in ["TIME", "MAG", "MAG_ERR", "TSTART", "EXPOSURE"]):
					continue

				time_sec = data["TSTART"] + data["EXPOSURE"] / 2
				mjd = MJDREF + time_sec / 86400.0
				filt = data.FILTER[0]

				color = filter_colors.get(filt.lower(), "gray")
				plt.errorbar(mjd, data["MAG"], yerr=data["MAG_ERR"], xerr = data["EXPOSURE"][0] / 86400,
					  fmt=".", color=color, label=filt)
	if len(obsids) ==1:
		print(data["EXPOSURE"])


	plt.gca().invert_yaxis()
	plt.xlabel("MJD")
	plt.ylabel("AB Magnitude")
	plt.title(f"Swift/UVOT Lightcurve — {transient}")
	plt.legend(fontsize=8)
	plt.tight_layout()

	outfile = os.path.join(plots_subdir, f"{transient}_lc.png")
	plt.savefig(outfile)
	plt.close()

	print(f"✅ Saved combined lightcurve for {transient} -> {outfile}")

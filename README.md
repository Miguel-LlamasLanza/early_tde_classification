# Early TDE clasiffication

Project containing the first steps on the TDE early photometric classification to be used within FINK, for ZTF and LSST data.

## Installation

Git clone this directory
Create virtual environment, and activate it. For instance:
```
conda create --name tdes python=3.10
conda activate tdes
```

In order to use the Rainbow fitting, we need to install the [light-curve-python package](https://github.com/light-curve/light-curve-python), for which we previously  need to install rust, with the following command:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Then, we can clone and pip install the [light-curve-python package](https://github.com/light-curve/light-curve-python):
```
python3 -mpip install 'light-curve[full]'
```

Then, simply install this project with
```
pip install -e .
```
## Usage

First, update the config.py file to match your preference. You need to update the paths.

The features can be extracted using the extract\_features.py script, where you can manually specify on which data you want to perform the feature extraction.
The script merge\_csvs.py provides can be used to put all the extracted features in the same csv.

Then, you can use the python script in the anomaly detection folder, in order to run the SNAD\_AAD method on the extracted features.

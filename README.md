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


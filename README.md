# Early TDE clasiffication

Project containing the first steps on the TDE early photometric classification to be used within FINK, for ZTF and LSST data.

## Installation

Git clone this directory
Create virtual environment, and activate it. For instance:
```
conda create --name tdes python=3.10
conda activate tdes
```

Install [ActSNClass](https://github.com/COINtoolbox/ActSNClass)
```
pip install git+https://github.com/COINtoolbox/ActSNClass
```

Install [ActSNFink](https://github.com/emilleishida/fink_sn_activelearning)

```
pip install git+https://github.com/emilleishida/fink_sn_activelearning@ee53bf21594c94b4bd4e6b4cbf706d0ca2c7c1c4
```
In order to install the [Rainbow package](https://github.com/erusseil/light-curve-python), we need to previously install rust.
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Clone and pip install the [Rainbow package](https://github.com/erusseil/light-curve-python).

Then, let’s install the oher dependencies needed.
```
pip install requests matplotlib pandas
```

For the Active Anomaly Detection notebook, you should install the [Coniferest](https://coniferest.readthedocs.io/en/latest/tutorial.html) package:

```
pip install coniferest
```

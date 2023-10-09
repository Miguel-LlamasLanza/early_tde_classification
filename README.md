# early_tde_classiffication

Project containing the first steps on the TDE early photometric classification to be used within FINK, for ZTF and LSST data.
It includes the repositories ActsnClass and the fink_sn_activelearning modules as subtrees. They are slightly modified to be adapted to our use-case. We added the chisq measure, to be consistent with the data we have from Zenodo.

## Installation


Create virtual environment, and activate it.
Git clone the repository and enter the directory. 

ActsnClass and the fink_sn_activelearning modules.
```
cd early_tde_classification/ActSNClass
pip install -r requirements.txt
python setup.py install

cd ../fink_sn_AL_classifier/
pip install -r requirements.txt
python setup.py install
```

Then, let’s install the oher dependencies needed.
```
pip install requests
pip install tqdm
```

# Early TDE clasiffication

Project containing the first steps on the TDE early photometric classification to be used within FINK, for ZTF and LSST data.
It includes the repositories ActsnClass and the fink_sn_activelearning modules as subtrees. They are slightly modified to be adapted to our use-case. We added the chisq measure, to be consistent with the data we have from Zenodo.

## Installation


Create virtual environment, and activate it.
Git clone the repository and enter the directory. 

Install [ActSNClass](https://github.com/COINtoolbox/ActSNClass)
```
pip install git+https://github.com/COINtoolbox/ActSNClass
```

Install [ActSNFink](https://github.com/emilleishida/fink_sn_activelearning)

```
pip install git+https://github.com/emilleishida/fink_sn_activelearning@ee53bf21594c94b4bd4e6b4cbf706d0ca2c7c1c4
```

Then, let’s install the oher dependencies needed.
```
pip install requests
pip install tqdm
```

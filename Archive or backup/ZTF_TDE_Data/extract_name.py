""" python extract_name.py <filename>
"""
import pandas as pd
import io
import sys
import requests
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
APIURL = 'https://fink-portal.org'

# grab filename
filename = sys.argv[1]

logging.info('Reading {}'.format(filename))

# Retrieve coordinates
with open(filename, 'r') as f:
    lines = f.readlines()
ra = float(lines[3].split(' = ')[1].split(' ')[0])
dec = float(lines[4].split(' = ')[1].split(' ')[0])

# Call fink conesearch
r = requests.post('{}/api/v1/explorer'.format(APIURL), json={'ra': ra, 'dec': dec, 'radius': 5.0, 'columns':'i:objectId'})

if r.json() == []:
    logging.warning('No counterpart in Fink.')
    sys.exit()

objectId = r.json()[0]['i:objectId']

r = requests.post('{}/api/v1/resolver'.format(APIURL), json={'resolver': 'tns', 'name': objectId, 'reverse': True})
print(r.json())

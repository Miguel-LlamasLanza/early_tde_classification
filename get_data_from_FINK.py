import requests
import pandas as pd
from io import BytesIO

# Get all objects falling within (center, radius) = ((ra, dec), radius)
# between 2021-06-25 05:59:37.000 (included) and 2021-07-01 05:59:37.000 (excluded)
r = requests.post(
  'https://fink-portal.org/api/v1/objects',
  json={
    'objectId': 'ZTF20acvezv'
  }
)

# Format output in a DataFrame
pdf = pd.read_json(BytesIO(r.content))

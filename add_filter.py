
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# Insert after Collimator S3 (index 9)
filter_s1 = {
    'name': 'Edge_Filter_1064_S1',
    'surface_type': 'Standard',
    'radius': 0.0,
    'thickness': 3.5,
    'glass': 'N-BK7',
    'conic': 0.0,
    'semi_diameter': 12.7,
    'is_coord_break': False
}

filter_s2 = {
    'name': 'Edge_Filter_1064_S2',
    'surface_type': 'Standard',
    'radius': 0.0,
    'thickness': 21.5, # 25.0 - 3.5
    'glass': 'AIR',
    'conic': 0.0,
    'semi_diameter': 12.7,
    'is_coord_break': False
}

# The previous surface (Collimator S3) needs its thickness adjusted so the filter is placed nicely, e.g. 10mm after collimator
data['surfaces'][9]['thickness'] = 10.0
# Then S1 thickness is 3.5
# Then S2 thickness is 25.0 - 10.0 - 3.5 = 11.5
filter_s2['thickness'] = 11.5

data['surfaces'].insert(10, filter_s1)
data['surfaces'].insert(11, filter_s2)

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


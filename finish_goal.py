
import sys, os, json, numpy as np
with open('spectrometer_yz_75mm_halfinch.json', 'r') as f:
    data = json.load(f)

# Change Collimator to 1/2 inch
collimator_indices = [i for i, s in enumerate(data['surfaces']) if 'Collimator' in s.get('name', '')]
for i, c_idx in enumerate(collimator_indices):
    data['surfaces'][c_idx]['semi_diameter'] = 6.35
    data['surfaces'][c_idx]['name'] = f'Collimator_AC127-030-C_S{i+1}'

# Set optimal distances
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
focus_s3 = focus_indices[-1]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
relay_s3 = relay_indices[-1]
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))

data['surfaces'][focus_s3]['thickness'] = 30.124
data['surfaces'][cb_idx]['thickness'] = 60.229
data['surfaces'][relay_s3]['thickness'] = 14.788
data['surfaces'][cond_idx+1]['thickness'] = 7.302

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


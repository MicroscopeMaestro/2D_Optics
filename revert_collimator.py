
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# Change Collimator back to 1 inch
collimator_indices = [i for i, s in enumerate(data['surfaces']) if 'Collimator' in s.get('name', '')]
for i, c_idx in enumerate(collimator_indices):
    data['surfaces'][c_idx]['semi_diameter'] = 12.7 # 1-inch radius
    data['surfaces'][c_idx]['name'] = f'Collimator_AC254-030-C_S{i+1}'

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


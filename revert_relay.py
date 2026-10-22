
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# Change Relay back to 1 inch
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
for i, r_idx in enumerate(relay_indices):
    data['surfaces'][r_idx]['semi_diameter'] = 12.7 # 1-inch radius
    data['surfaces'][r_idx]['name'] = f'Relay_AC254-075-C_S{i+1}'

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


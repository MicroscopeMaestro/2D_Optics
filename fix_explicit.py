
import sys, os, json, numpy as np

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)

# Collimator: AC254-030-C
data['surfaces'][6]['thickness'] = 25.5 # Slit to Collimator R3 (FFL of 30mm doublet)

# Focusing: change to AC254-050-C
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
r1, r2, r3 = 34.5, -23.1, -135.2
t1, t2 = 12.0, 2.0
data['surfaces'][focus_indices[0]]['radius'] = r1
data['surfaces'][focus_indices[0]]['thickness'] = t1
data['surfaces'][focus_indices[0]]['glass'] = 'N-BAF10'
data['surfaces'][focus_indices[1]]['radius'] = r2
data['surfaces'][focus_indices[1]]['thickness'] = t2
data['surfaces'][focus_indices[1]]['glass'] = 'N-SF6HT'
data['surfaces'][focus_indices[2]]['radius'] = r3
for i in range(3): data['surfaces'][focus_indices[i]]['name'] = f'Focusing_AC254-050-C_S{i+1}'

data['surfaces'][focus_indices[2]]['thickness'] = 46.3 # BFL of 50mm doublet

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


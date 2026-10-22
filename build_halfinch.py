
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_yz_75mm_stop4.json', 'r') as f:
    data = json.load(f)

relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
for i, r_idx in enumerate(relay_indices):
    data['surfaces'][r_idx]['semi_diameter'] = 6.35 # 1/2-inch
    data['surfaces'][r_idx]['name'] = f'Relay_AC127-075-C_S{i+1}'

with open('spectrometer_yz_75mm_halfinch.json', 'w') as f:
    json.dump(data, f, indent=4)


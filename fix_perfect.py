
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)

# True AC254-030-C radii (from Collect lens)
r1, r2, r3 = 19.4, -14.2, -64.3
t1, t2 = 7.0, 2.0

collimator_indices = [i for i, s in enumerate(data['surfaces']) if 'Collimator' in s.get('name', '')]
data['surfaces'][collimator_indices[0]]['radius'] = -r3
data['surfaces'][collimator_indices[0]]['thickness'] = t2
data['surfaces'][collimator_indices[0]]['glass'] = 'F2'
data['surfaces'][collimator_indices[1]]['radius'] = -r2
data['surfaces'][collimator_indices[1]]['thickness'] = t1
data['surfaces'][collimator_indices[1]]['glass'] = 'N-BAF10'
data['surfaces'][collimator_indices[2]]['radius'] = -r1

focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
data['surfaces'][focus_indices[0]]['radius'] = r1
data['surfaces'][focus_indices[0]]['thickness'] = t1
data['surfaces'][focus_indices[0]]['glass'] = 'N-BAF10'
data['surfaces'][focus_indices[1]]['radius'] = r2
data['surfaces'][focus_indices[1]]['thickness'] = t2
data['surfaces'][focus_indices[1]]['glass'] = 'F2'
data['surfaces'][focus_indices[2]]['radius'] = r3

data['surfaces'][6]['thickness'] = 23.3 # Slit to Collimator
data['surfaces'][focus_indices[2]]['thickness'] = 23.3 # Focusing to DMD

dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
data['surfaces'][cb_idx]['thickness'] = 71.0 # DMD to Relay

relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
data['surfaces'][relay_indices[0]]['radius'] = 57.99
data['surfaces'][relay_indices[1]]['radius'] = -46.39
data['surfaces'][relay_indices[2]]['radius'] = -144.97
data['surfaces'][relay_indices[2]]['thickness'] = 10.0 # Relay to Condenser

cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
data['surfaces'][cond_idx+1]['thickness'] = 7.3 # Condenser to detector

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


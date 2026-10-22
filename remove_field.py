
import sys, os, json, copy, numpy as np
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

new_surfaces = []
for s in data['surfaces']:
    if not s.get('name', '').startswith('Field_'):
        new_surfaces.append(s)
    else:
        # Transfer the thickness of Field_B to the previous surface (DMD) to maintain total distance!
        if s.get('name', '') == 'Field_B':
            new_surfaces[-1]['thickness'] += s['thickness']

data['surfaces'] = new_surfaces
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Removed Field lens!')


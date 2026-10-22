
import sys, os, json

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

# Find old stop and remove it
old_stop_idx = -1
for i, s in enumerate(data['surfaces']):
    if s.get('name') == 'Aperture_Stop':
        old_stop_idx = i
        break

if old_stop_idx != -1:
    del data['surfaces'][old_stop_idx]
    # Find M1 and restore thickness to 150.0
    for s in data['surfaces']:
        if 'M1_Collimator' in s.get('name', ''):
            s['thickness'] = 150.0
            break

# Create new stop
new_stop = {
    'name': 'Aperture_Stop',
    'surface_type': 'Stop',
    'thickness': 0.0, # Exactly at the first lens
    'radius': 0.0,
    'glass': 'AIR',
    'aperture_type': 'Circular',
    'semi_diameter': 3.0, # 6 mm diameter
    'decenter_y': 0.0,
    'tilt_x': 0.0
}

# Insert at index 1 (right after Object_Plane, before Fore_L1_F)
data['surfaces'].insert(1, new_stop)

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'w') as f: json.dump(data, f, indent=4)
print('Moved stop to collection optics!')


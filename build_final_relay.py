
import sys, os, json

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'DMD' in s.get('name', ''):
        dmd_idx = i
        break

data['surfaces'] = data['surfaces'][:dmd_idx+1]
data['surfaces'][dmd_idx]['thickness'] = 200.0

# Add new relay L1 (AC254-200-C)
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_F', 'surface_type': 'Standard', 'radius': 129.8, 'thickness': 3.6, 'glass': 'N-BK7', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_M', 'surface_type': 'Standard', 'radius': -93.8, 'thickness': 2.0, 'glass': 'SF11', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_B', 'surface_type': 'Standard', 'radius': -342.3, 'thickness': 230.0, 'glass': 'AIR', 'semi_diameter': 12.7})

# Add new relay L2 (AC254-030-C)
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_F', 'surface_type': 'Standard', 'radius': 19.9, 'thickness': 9.0, 'glass': 'N-BK7', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_M', 'surface_type': 'Standard', 'radius': -15.3, 'thickness': 2.0, 'glass': 'SF11', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_B', 'surface_type': 'Standard', 'radius': -55.7, 'thickness': 27.5, 'glass': 'AIR', 'semi_diameter': 12.7})

# Add detector
data['surfaces'].append({'name': 'Image_Detector', 'surface_type': 'Detector', 'radius': 0.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 0.4})

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'w') as f: json.dump(data, f, indent=4)


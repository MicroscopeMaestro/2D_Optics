
import sys, os, json

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'r') as f: data = json.load(f)

for s in data['surfaces']:
    name = s.get('name', '')
    if 'Fore_L1' in name:
        s['name'] = name.replace('Fore_L1', 'AC254-050-C_Fore_L1')
    elif 'Fore_L2' in name:
        s['name'] = name.replace('Fore_L2', 'AC254-050-C_Fore_L2')
    elif 'M1_Collimator' in name:
        s['name'] = 'CM254-050-E04_M1_Collimator'
    elif 'Diffraction_Grating' in name:
        s['name'] = 'GR25-0310_Diffraction_Grating'
    elif 'Relay_L1' in name:
        s['name'] = name.replace('Relay_L1', 'AC254-100-C_Relay_L1')
    elif 'Relay_L2' in name:
        s['name'] = name.replace('Relay_L2', 'ACL25416U_Relay_L2')

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'w') as f: json.dump(data, f, indent=4)
print('Renamed surfaces.')


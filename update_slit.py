
import json
import os

filepath = 'teamwork_yz_spectrometer/spectrometer_yz_1064.json'
with open(filepath, 'r') as f:
    data = json.load(f)

for s in data['surfaces']:
    if 'Slit' in s.get('name', ''):
        s['name'] = 'Input_Slit_200um'
        s['semi_diameter'] = 0.1

with open(filepath, 'w') as f:
    json.dump(data, f, indent=4)

# Also update the labeled one in artifacts
filepath_artifacts = r'C:\Users\Juan\.gemini\antigravity\brain\49fd158d-b0e1-421c-b15b-b237b2e3ad7f\teamwork_yz_1064_labeled.json'
if os.path.exists(filepath_artifacts):
    with open(filepath_artifacts, 'r') as f:
        data_art = json.load(f)
    for s in data_art['surfaces']:
        if 'Slit' in s.get('name', ''):
            s['name'] = 'Input_Slit_200um'
            s['semi_diameter'] = 0.1
    with open(filepath_artifacts, 'w') as f:
        json.dump(data_art, f, indent=4)

print('Updated slit to 200um (0.1mm radius) in both JSON files.')


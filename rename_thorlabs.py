
import json
import os

filepath = 'teamwork_yz_spectrometer/spectrometer_yz_1064.json'
with open(filepath, 'r') as f:
    data = json.load(f)

# Define the replacements
replacements = {
    'Collect_L1': 'Collect_ACL25416U',
    'Collect_L2': 'Collect_AC254-030-C',
    'Collimator': 'Collimator_AC254-030-C',
    'Focusing': 'Focusing_AC254-050-C',
    'Imaging_Relay': 'Relay_AC254-050-C',
    'Imaging_Cond': 'Condenser_ACL25416U'
}

for s in data['surfaces']:
    name = s.get('name', '')
    for old, new in replacements.items():
        if name.startswith(old):
            s['name'] = name.replace(old, new)

# Save the updated JSON
with open(filepath, 'w') as f:
    json.dump(data, f, indent=4)

print('Renamed elements to include Thorlabs part numbers.')


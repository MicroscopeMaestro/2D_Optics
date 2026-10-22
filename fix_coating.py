
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)

for s in data['surfaces']:
    if 'ACL25416U_S' in s.get('name', ''):
        s['name'] = s['name'].replace('ACL25416U', 'ACL25416U-C')

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)


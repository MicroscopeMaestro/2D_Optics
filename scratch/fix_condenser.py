import json
filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

for s in config['surfaces']:
    if s['name'] == 'Condenser_ACL25416U_S2':
        s['thickness'] = 10.13
        break

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Updated Condenser_ACL25416U_S2 thickness to 10.13 mm")

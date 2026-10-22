import json
filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Find CB_After_DMD and update its thickness
for s in config['surfaces']:
    if s['name'] == 'CB_After_DMD':
        s['thickness'] = 74.76
        break

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Updated CB_After_DMD thickness to 74.76 mm")

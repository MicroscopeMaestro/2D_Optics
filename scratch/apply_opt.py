import json

filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Apply optimized distances
config['surfaces'][17]['thickness'] = 43.80
config['surfaces'][18]['thickness'] = 97.03
config['surfaces'][22]['thickness'] = 19.88
config['surfaces'][24]['thickness'] = 8.525

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Applied final optimized distances.")

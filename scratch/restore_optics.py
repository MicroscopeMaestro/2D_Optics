import json

filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Restore AC254-050-C focusing lens
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][15]['name'] = 'Focusing_AC254-050-C_S1'

config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][16]['name'] = 'Focusing_AC254-050-C_S2'

config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80 # Keep standard infinite conjugate focus for now
config['surfaces'][17]['name'] = 'Focusing_AC254-050-C_S3'

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

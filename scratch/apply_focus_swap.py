import json

filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Change Focusing lens to AC254-030-C
config['surfaces'][15]['radius'] = 20.3
config['surfaces'][15]['thickness'] = 7.0
config['surfaces'][15]['glass'] = 'N-BAF10'
config['surfaces'][15]['name'] = 'Focusing_AC254-030-C_S1'

config['surfaces'][16]['radius'] = -16.4
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][16]['name'] = 'Focusing_AC254-030-C_S2'

config['surfaces'][17]['radius'] = -63.7
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 23.11 # Optimized
config['surfaces'][17]['name'] = 'Focusing_AC254-030-C_S3'

# Reset Relay and Condenser to standard infinite conjugate
config['surfaces'][18]['thickness'] = 74.76
config['surfaces'][22]['thickness'] = 28.0
config['surfaces'][24]['thickness'] = 10.13

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Applied AC254-030-C focusing lens.")

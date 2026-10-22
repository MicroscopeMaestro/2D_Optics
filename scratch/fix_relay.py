import json
import os

filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Relay is indices 20, 21, 22
s1 = config['surfaces'][20]
s2 = config['surfaces'][21]
s3 = config['surfaces'][22]

old_s3_t = s3['thickness']

s1['radius'] = 270.4
s1['thickness'] = 2.0
s1['glass'] = 'N-SF6HT'

s2['radius'] = 44.2
s2['thickness'] = 4.0
s2['glass'] = 'N-BAF10'

s3['radius'] = -51.5
s3['thickness'] = old_s3_t
s3['glass'] = 'AIR'

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Flipped Relay_AC254-100-C orientation.")

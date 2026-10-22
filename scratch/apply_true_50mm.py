import json
filename = 'c:/Users/Juan/Desktop/spectrometer_yz_40mm_100mm_slit.json'

with open(filename, 'r') as f:
    config = json.load(f)

# True AC254-050-C Prescription (from Thorlabs)
config['surfaces'][15]['radius'] = 22.9
config['surfaces'][15]['thickness'] = 9.0
config['surfaces'][15]['glass'] = 'N-LAK22'
config['surfaces'][15]['name'] = 'Focusing_AC254-050-C_S1'

config['surfaces'][16]['radius'] = -25.9
config['surfaces'][16]['thickness'] = 1.8
config['surfaces'][16]['glass'] = 'N-SF6'
config['surfaces'][16]['name'] = 'Focusing_AC254-050-C_S2'

config['surfaces'][17]['radius'] = 194.5
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 41.2
config['surfaces'][17]['name'] = 'Focusing_AC254-050-C_S3'

# We MUST increase the semi_diameter of the DMD because 5.0 clips the spectrum!
config['surfaces'][18]['semi_diameter'] = 25.0

# Optimized back-end distances for demagnification
config['surfaces'][18]['thickness'] = 55.02
config['surfaces'][22]['thickness'] = 26.00
config['surfaces'][24]['thickness'] = 7.61

# Restore apertures to standard
for s in config['surfaces'][19:]:
    if 'aperture_y_half' in s:
        s['aperture_y_half'] = 15.0 # CB has 15
        s['aperture_x_half'] = 20.0
        s['semi_diameter'] = 12.7

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Applied True AC254-050-C and new distances to Desktop JSON.")

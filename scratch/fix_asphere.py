import json

filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# S1 is index 1
s1 = config['surfaces'][1]
# S2 is index 2
s2 = config['surfaces'][2]

# Swap their shapes!
# Make S1 flat
s1['surface_type'] = 'Standard'
s1['radius'] = 0.0
s1['conic'] = 0.0
s1['asphere_coeffs'] = []

# Make S2 aspheric (light exiting convex surface -> radius is negative)
s2['surface_type'] = 'Aspheric'
s2['radius'] = -8.32
s2['conic'] = -1.0
s2['asphere_coeffs'] = []

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Fixed the orientation of the first aspheric lens.")


import sys, os, json

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'r') as f: data = json.load(f)

# Ensure Object is named properly
data['surfaces'][0]['name'] = 'Object_Plane'

# Ensure Image is named properly
data['surfaces'][-1]['name'] = 'Image_Detector'

# Find M1 Collimator
m1_idx = -1
for i, s in enumerate(data['surfaces']):
    if 'M1_Collimator' in s['name']:
        m1_idx = i
        break

# M1 has thickness 150.0. We will split this.
# Let's place the STOP exactly at the Grating (thickness 150.0 for M1, 0.0 for Stop)
# Wait, if thickness is 0.0, it's co-located with the Grating. 
# In Zemax, placing the stop at the grating is done by making the Grating the STOP surface.
# But if the user wants 3 explicit surfaces (Object, Stop, Image), they might want a dedicated dummy surface!
# Let's put the Stop 50 mm before the Grating. So M1 -> 100mm -> Stop -> 50mm -> Grating.
data['surfaces'][m1_idx]['thickness'] = 100.0

stop_surface = {
    'name': 'Aperture_Stop',
    'surface_type': 'Standard',
    'thickness': 50.0,
    'radius': 0.0,
    'glass': 'AIR',
    'aperture_type': 'Circular',
    'semi_diameter': 5.0, # 10mm stop diameter
    'decenter_y': 0.0,
    'tilt_x': 0.0
}

data['surfaces'].insert(m1_idx + 1, stop_surface)

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved Zemax layout!')


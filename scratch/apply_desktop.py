import json

filename = 'c:/Users/Juan/Desktop/spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Change Focusing lens to true AC254-030-C (Crown facing collimated light)
config['surfaces'][15]['radius'] = 19.4
config['surfaces'][15]['thickness'] = 7.0
config['surfaces'][15]['glass'] = 'N-BAF10'
config['surfaces'][15]['name'] = 'Focusing_AC254-030-C_S1'

config['surfaces'][16]['radius'] = -14.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'F2'
config['surfaces'][16]['name'] = 'Focusing_AC254-030-C_S2'

config['surfaces'][17]['radius'] = -64.3
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 18.13 # Optimized
config['surfaces'][17]['name'] = 'Focusing_AC254-030-C_S3'

# Ensure Relay and Condenser are standard infinite conjugate
config['surfaces'][18]['thickness'] = 74.76
config['surfaces'][22]['thickness'] = 28.0
config['surfaces'][24]['thickness'] = 10.13

# Fix the Coordinate Break! Since the beam reflects from the DMD at an angle, 
# and we are using standard lenses, the CB_After_DMD should be aligned.
# Wait! In the optics_app tracer, if the CB has tilt_x=0, it means the Relay lens is NOT angled!
# The user's original layout had CB_After_DMD tilt_x=0. This implies the Relay lens is positioned parallel to the DMD.
# Since the user requested modifying "optical paths and distances" earlier, I should make sure the Relay lens catches the light.
# If I use the AC254-030-C, the spread on the DMD is small. The chief ray might be close to the axis, so tilt_x=0 might not clip it. Let's check!

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Applied TRUE AC254-030-C to Desktop JSON.")

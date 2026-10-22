import json
with open('fiber_spectrometer_setup.json') as f:
    d = json.load(f)
for i, s in enumerate(d['stack']['surfaces']):
    print(f'Surf {i} tilts: {s["tilt_x"]}, {s["tilt_y"]}, {s["tilt_z"]}')

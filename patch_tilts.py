import json

with open('spectrometer_raman_50mm_100mm_manual.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for s in data['surfaces']:
    if s['name'] in ['Air Gap (Grating)', 'Air Gap (DMD)']:
        s['tilt_x'] = 0.0
        s['tilt_y'] = 0.0
        s['tilt_z'] = 0.0

with open('spectrometer_raman_50mm_100mm_manual.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Tilts fixed")

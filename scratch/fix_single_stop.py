import json
with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)

for s in d['surfaces']:
    name = s.get('name', '')
    if 'Entrance_Slit' in name:
        s['surface_type'] = 'Standard'
        s['glass'] = 'AIR'
    elif 'Aperture_Stop' in name:
        s['surface_type'] = 'Stop'

with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'w') as f:
    json.dump(d, f, indent=4)
print('Fixed stop vs standard.')

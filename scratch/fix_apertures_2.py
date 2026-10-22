import json
with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)

for s in d['surfaces']:
    name = s.get('name', '')
    if 'Slit' in name:
        s['semi_diameter'] = 12.7
        s['aperture_x_half'] = 0.5
        s['aperture_y_half'] = 0.2
    elif 'Aperture' in name:
        s['semi_diameter'] = 12.7
        s['aperture_type'] = 'Circular'
        s['aperture_x_half'] = 0.0  # Not rectangular, clear out unused params
        s['aperture_y_half'] = 0.0
        s['aperture_r_outer'] = 12.7 # Hole radius 12.7 -> 25.4mm hole

with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'w') as f:
    json.dump(d, f, indent=4)
print('Fixed apertures.')

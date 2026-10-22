import json
with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)

for s in d['surfaces']:
    if 'Aperture' in s['name'] or 'Slit' in s['name']:
        s['semi_diameter'] = 12.7
        # For the drawn bounding box in CAD:
        if s.get('aperture_x_half', 0) > 0:
            s['aperture_x_half'] = 12.7
        if s.get('aperture_y_half', 0) > 0:
            # Leave the slit hole alone!
            if 'Slit' not in s['name']:
                s['aperture_y_half'] = 12.7

with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'w') as f:
    json.dump(d, f, indent=4)
print('Updated apertures for real CAD size.')

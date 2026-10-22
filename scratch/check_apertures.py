import json
d = json.load(open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'))
for i, s in enumerate(d['surfaces']):
    if 'Aperture' in s['name'] or 'Slit' in s['name']:
        print(f"{s['name']:15} -> semi: {s.get('semi_diameter')}, type: {s.get('aperture_type')}, x: {s.get('aperture_x_half')}, y: {s.get('aperture_y_half')}, r_out: {s.get('aperture_r_outer')}")

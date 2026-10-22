import json
d = json.load(open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'))
for s in d['surfaces']:
    name = s.get('name', '')
    if 'Aperture' in name or 'Slit' in name:
        print(f"{name}: semi={s.get('semi_diameter')}, type={s.get('aperture_type')}, ap_x={s.get('aperture_x_half')}, ap_r_out={s.get('aperture_r_outer')}")

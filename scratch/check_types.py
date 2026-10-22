import json
d = json.load(open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'))
for i, s in enumerate(d['surfaces']):
    if 'Aperture' in s['name'] or 'Slit' in s['name']:
        print(f"{s['name']:15} -> surface_type: {s.get('surface_type')}")

import json
with open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)

for s in d['surfaces']:
    if 'Slit' in s.get('name', ''):
        print('Name:', s.get('name'))
        print('Type:', s.get('surface_type'))
        print('Ap Type:', s.get('aperture_type'))
        print('Ap X:', s.get('aperture_x_half'))
        print('Ap Y:', s.get('aperture_y_half'))
        print('Glass:', s.get('glass'))

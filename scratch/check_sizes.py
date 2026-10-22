import json
d = json.load(open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'))
for i, s in enumerate(d['surfaces']):
    if 'Aperture' in s['name'] or 'AC254' in s['name'] or 'ACL254' in s['name']:
        sd = s.get('semi_diameter')
        ar = s.get('aperture_r_outer', s.get('aperture_x_half'))
        print(f"{s['name']:<20} - semi_diam: {sd}, aperture_param: {ar}")

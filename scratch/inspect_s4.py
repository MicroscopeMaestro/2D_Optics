import json
with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)
s4 = d['surfaces'][4]
print("S4:", s4.get('name'), "type:", s4.get('surface_type'), "is_stop:", s4.get('is_stop'), "sd:", s4.get('semi_diameter'), "ap_type:", s4.get('aperture_type'), "ap_x:", s4.get('aperture_x_half'), "ap_y:", s4.get('aperture_y_half'))

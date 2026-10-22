import json
with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)
surfs = d.get('surfaces', [])
s15 = surfs[15]
s20 = surfs[20]
print("S15 semi_diameter:", s15.get('semi_diameter'), "ap_x:", s15.get('aperture_x_half'), "ap_y:", s15.get('aperture_y_half'))
print("S20 semi_diameter:", s20.get('semi_diameter'), "ap_x:", s20.get('aperture_x_half'), "ap_y:", s20.get('aperture_y_half'))

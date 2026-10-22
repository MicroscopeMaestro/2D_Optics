import json
with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)
s20 = d['surfaces'][20]
print("S20:", s20.get('name'), "t:", s20.get('thickness'), "type:", s20.get('surface_type'))

import json
d = json.load(open('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'))
for i, s in enumerate(d['surfaces']):
    print(f"{i}: {s.get('name')}")

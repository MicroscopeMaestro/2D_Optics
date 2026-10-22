import json
with open('spectrometer_yz_40mm_100mm_optimized.json') as f:
    d = json.load(f)
for i, s in enumerate(d['surfaces']):
    print(f"{i}: {s['name']} (thick: {s['thickness']})")

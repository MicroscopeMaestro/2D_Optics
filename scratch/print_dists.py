import json

with open('spectrometer_raman_50mm_100mm.json') as f:
    config = json.load(f)

for i, s in enumerate(config['surfaces']):
    print(f"{i:02d}: {s['name']:30s} -> thickness to next: {s['thickness']:>6.2f} mm")

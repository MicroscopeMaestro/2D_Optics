import json
with open('spectrometer_yz_40mm_100mm_final.json', 'r') as f:
    config = json.load(f)

print("Surface Thicknesses:")
for i, s in enumerate(config['surfaces']):
    print(f"  {i}: {s['name']} (thick={s['thickness']})")

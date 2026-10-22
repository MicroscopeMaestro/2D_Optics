import json
data = json.load(open('spectrometer_yz_40mm_100mm_optimized.json'))
for s in data['surfaces']:
    print(f"{s['name']:30} {s.get('semi_diameter', 0)}")

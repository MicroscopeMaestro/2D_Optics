import json

with open('spectrometer_raman_50mm_100mm_manual.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for s in data['surfaces']:
    if s['name'] == 'Air Gap (DMD)':
        s['name'] = 'Air Gap 2'
    elif s['name'] == 'Air Gap (Grating)':
        s['name'] = 'Air Gap 1'

with open('spectrometer_raman_50mm_100mm_manual.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Renamed")

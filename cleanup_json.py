import json

with open('spectrometer_raman_50mm_100mm_manual.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_surfaces = []
seen = set()

for s in data['surfaces']:
    if s['name'] == 'Air Gap (Grating)':
        if s['thickness'] == 0.0:
            continue # skip the duplicate 0 thickness one
    if s['name'] == 'Air Gap (DMD)':
        if s['thickness'] == 0.0:
            continue # skip the duplicate 0 thickness one
    new_surfaces.append(s)

data['surfaces'] = new_surfaces

with open('spectrometer_raman_50mm_100mm_manual.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Duplicates removed")

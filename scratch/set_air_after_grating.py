import json

path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

# Surface 15 is Grating: 0 thickness!
s15 = d['surfaces'][15]
s15['thickness'] = 0.0

# Surface 16 is named and typed as AIR_SPACE
s16 = d['surfaces'][16]
s16.pop('is_air_space', None) # remove non-field key
s16['name'] = 'Air_after_grating'
s16['surface_type'] = 'Air Space'
s16['glass'] = 'AIR'
s16['thickness'] = 20.0
s16['radius'] = 0.0

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4)

print("Updated desktop layout: Grating thickness = 0.0, Air_after_grating (Air Space) = 20.0")

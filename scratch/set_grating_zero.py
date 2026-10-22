import json

path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

# Surface 15 is Grating
s15 = d['surfaces'][15]
s15['thickness'] = 0.0

# Surface 16 is S16
s16 = d['surfaces'][16]
s16['is_air_space'] = True
s16['thickness'] = 20.0

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4)

print("Grating thickness set to 0.0, S16 marked as is_air_space = True with t = 20.0")

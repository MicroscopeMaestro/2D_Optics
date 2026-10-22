import json

path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

d['surfaces'][15]['thickness'] = 10.0
d['surfaces'][16]['thickness'] = 10.0

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4)

print("Restored S15.thickness = 10.0 and S16.thickness = 10.0 in desktop layout!")

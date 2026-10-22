import json

path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

s15 = d['surfaces'][15]
s16 = d['surfaces'][16]
s20 = d['surfaces'][20]

print("S15:", s15['name'], "t:", s15['thickness'])
print("S16:", s16['name'], "t:", s16['thickness'])
print("S20:", s20['name'], "t:", s20['thickness'])

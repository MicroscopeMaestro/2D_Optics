import json

path = r'C:\Users\Juan\Desktop\.history\spectrometer_raman_50mm_100mm_manual_optimized_v008_20261010_120816.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for i in [14, 15, 16, 17, 18, 19, 20, 21]:
    s = d['surfaces'][i]
    print(f"[{i:02d}] {s.get('name', ''):<25} | t: {s.get('thickness', 0):<6.2f} | glass: {s.get('glass', ''):<8} | type: {s.get('surface_type', ''):<10}")

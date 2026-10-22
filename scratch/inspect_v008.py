import json

path = r'C:\Users\Juan\Desktop\.history\spectrometer_raman_50mm_100mm_manual_optimized_v008_20261010_120816.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

surfs = d.get('surfaces', [])
print(f"Original v008 surface count: {len(surfs)}")
for i, s in enumerate(surfs):
    print(f"[{i:02d}] {s.get('name', ''):<25} | type: {s.get('surface_type', ''):<10} | glass: {s.get('glass', ''):<8} | t: {s.get('thickness', 0):<6.2f} | sd: {s.get('semi_diameter', 0):<5.2f}")

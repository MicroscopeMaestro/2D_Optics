import json
path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for i, s in enumerate(d.get('surfaces', [])):
    print(f"[{i:02d}] {s.get('name', ''):<28} | t={s.get('thickness', 0):<6.2f} | glass={s.get('glass', ''):<8} | type={s.get('surface_type', ''):<10} | sd={s.get('semi_diameter', 0):<5.2f} | shape={s.get('substrate_shape', '')} | ap={s.get('aperture_type', '')}")

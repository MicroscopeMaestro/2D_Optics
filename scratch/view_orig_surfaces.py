import json

with open('spectrometer_raman_50mm_100mm_manual.original_2026-10-04.json', 'r') as f:
    d = json.load(f)

for i, s in enumerate(d['surfaces']):
    cat = s.get('catalog_metadata', {})
    print(f"{i:2d}: {s['name']:<25} t={s.get('thickness',0):<6} glass={s.get('glass'):<10} stype={s.get('surface_type')} cat={cat}")

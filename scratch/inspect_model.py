import json
from optics_app.catalogs.thorlabs import get_catalog_lens

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    d = json.load(f)

print(f"Version: {d.get('version')}")
print(f"Num surfaces: {len(d.get('surfaces', []))}")
print(f"Num components: {len(d.get('components', []))}")

for i, s in enumerate(d.get('surfaces', [])):
    print(f"{i:2d}: {s.get('name'):<25} grp={s.get('group_name',''):<16} R={s.get('radius',0):<10.3f} t={s.get('thickness',0):<8.3f} glass={s.get('glass'):<10} type={s.get('surface_type')}")

print("\n--- Components ---")
for c in d.get('components', []):
    print(f"UID: {c.get('uid')} | Name: {c.get('name')} | Part: {c.get('catalog_part')} | Surfaces: {c.get('surface_indices')}")

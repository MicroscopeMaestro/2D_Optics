import json, copy

src = r'C:\Users\Juan\Desktop\.history\spectrometer_raman_50mm_100mm_manual_optimized_v008_20261010_120816.json'
dst = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'

with open(src, 'r', encoding='utf-8') as f:
    d = json.load(f)

# Keep the exact 24 surfaces and all exact thickness and coordinate positions
# S15 had semi_diameter: 0.0 with aperture_x/y = 12.5. Give it its proper clear semi_diameter (17.68)
d['surfaces'][15]['semi_diameter'] = 17.6777

with open(dst, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4)

print("Restored exact 24 surfaces to Desktop layout:")
for i, s in enumerate(d['surfaces']):
    print(f"[{i:02d}] {s.get('name', ''):<25} | t: {s.get('thickness', 0):<6.2f} | glass: {s.get('glass', ''):<8} | sd: {s.get('semi_diameter', 0):<5.2f}")

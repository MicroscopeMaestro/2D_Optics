import json, os, glob

def inspect(name):
    with open(name) as f:
        d = json.load(f)
    print(f"=== {name} ===")
    print("Version:", d.get("version"))
    print("Wavelengths:", d.get("wavelengths"))
    print("Source Config:", d.get("source_config"))
    print(f"{'Idx':<4} {'Name':<12} {'Type':<10} {'Radius':<9} {'Thick':<9} {'Glass':<10} {'SD':<6} {'TiltX':<7} {'TiltY':<7} {'TiltZ':<7} {'Grating':<10}")
    for i, s in enumerate(d.get("surfaces", [])):
        gr = f"{s.get('grating_lines_per_mm',0)} l/mm" if s.get("surface_type") == "Grating" else ""
        print(f"{i:<4} {s.get('name',''):<12} {s.get('surface_type',''):<10} {s.get('radius',0):<9.2f} {s.get('thickness',0):<9.2f} {s.get('glass',''):<10} {s.get('semi_diameter',0):<6.2f} {s.get('tilt_x',0):<7.2f} {s.get('tilt_y',0):<7.2f} {s.get('tilt_z',0):<7.2f} {gr:<10}")

inspect("hybrid_raman_1064_full_range.json")
print()
inspect("hybrid_raman_1064_physically_correct.json")
print()
inspect("hybrid_raman_1064_fingerprint.json")

import json
import glob

files = glob.glob("hybrid_raman*.json")
for fname in sorted(files):
    print("=" * 70)
    print("FILE:", fname)
    with open(fname, "r", encoding="utf-8") as f:
        d = json.load(f)
    print("Version:", d.get("version"))
    print("Wavelengths:", d.get("wavelengths"))
    print("Primary:", d.get("primary_wavelength"))
    print("Source config:", d.get("source_config"))
    surfaces = d.get("surfaces", [])
    print(f"Surfaces count: {len(surfaces)}")
    for i, s in enumerate(surfaces):
        print(f"  [{i:2d}] {s.get('name', ''):30s} | Type: {s.get('surface_type', ''):10s} | R: {s.get('radius', 0):8.2f} | T: {s.get('thickness', 0):6.2f} | Glass: {s.get('glass', ''):10s} | TiltY: {s.get('tilt_y', 0):5.1f} | Lines: {s.get('grating_lines_per_mm', 0)}")

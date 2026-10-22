import json

for fname in ['hybrid_raman_1064.json', 'hybrid_raman_1064_full_range.json', 'hybrid_raman_1064_physically_correct.json', 'hybrid_raman_1064_fingerprint.json']:
    print('=' * 60)
    print(fname)
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            d = json.load(f)
        print("Version:", d.get("version"))
        print("Wavelengths:", d.get("wavelengths"))
        print("Primary:", d.get("primary_wavelength"))
        print("Source config:", d.get("source_config"))
        for i, s in enumerate(d.get('surfaces', [])):
            print(f"  [{i:2d}] {s.get('name', ''):30s} | Type: {s.get('surface_type', ''):10s} | R: {s.get('radius', 0):8.2f} | T: {s.get('thickness', 0):6.2f} | Glass: {s.get('glass', ''):10s} | TiltY: {s.get('tilt_y', 0):5.1f} | SD: {s.get('semi_diameter', 0):5.2f}")
    except Exception as e:
        print("Error:", e)

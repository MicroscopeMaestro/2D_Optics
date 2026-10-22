import json, glob

files = [
    'thorlabs_all_mirror_spectrometer.json',
    'dmd4500_mirror_spectrometer_785nm.json',
    'dmd4500_10_to_1_mirror_spectrometer_app_model.json',
    'thorlabs_czerny_turner_785nm.json'
]

for fname in files:
    try:
        with open(fname) as f:
            d = json.load(f)
        print(f"=== {fname} ===")
        print("Wavelengths:", d.get("wavelengths"))
        print("Primary wavelength:", d.get("primary_wavelength"))
        for i, s in enumerate(d.get("surfaces", [])):
            print(f"  {i:2d}: {s.get('name'):<35} | Type: {str(s.get('surface_type')):<10} | R: {str(s.get('radius')):<8} | T: {str(s.get('thickness')):<8} | Glass: {str(s.get('glass')):<8} | TiltX: {str(s.get('tilt_x')):<6} | TiltY: {str(s.get('tilt_y')):<6} | DecY: {str(s.get('decenter_y')):<6} | Semi: {str(s.get('semi_diameter')):<6} | Lines: {s.get('grating_lines_per_mm')}")
        print()
    except Exception as e:
        print(f"Error {fname}: {e}")

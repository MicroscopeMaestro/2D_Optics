import json

with open('Spectrometer_V2_Perfected.json') as f:
    d = json.load(f)

print('Wavelengths:', d.get('wavelengths'))
print('Primary wavelength:', d.get('primary_wavelength'))
print('Surfaces count:', len(d.get('surfaces', [])))
for i, s in enumerate(d.get('surfaces', [])):
    print(f"{i:2d}: {s.get('name'):<30} | Type: {str(s.get('surface_type')):<10} | R: {s.get('radius'):<8} | T: {s.get('thickness'):<8} | Glass: {str(s.get('glass')):<12} | TiltX: {s.get('tilt_x'):<6} | TiltY: {s.get('tilt_y'):<6} | DecY: {s.get('decenter_y'):<6} | Semi: {s.get('semi_diameter'):<6} | Lines: {s.get('grating_lines_per_mm')}")

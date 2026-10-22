import json
with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)
for i, s in enumerate(d.get('surfaces', [])):
    if s.get('surface_type') == 'Grating':
        print(f"Before S{i}: semi_diameter={s.get('semi_diameter')}, ap_x={s.get('aperture_x_half')}, ap_y={s.get('aperture_y_half')}")
        if s.get('semi_diameter', 0.0) <= 0.0:
            hx = s.get('aperture_x_half', 12.5)
            hy = s.get('aperture_y_half', 12.5)
            import math
            s['semi_diameter'] = round(math.sqrt(hx**2 + hy**2), 4)
            print(f"After S{i}: semi_diameter={s.get('semi_diameter')}")

with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'w') as f:
    json.dump(d, f, indent=4)
print("Updated Desktop model successfully.")

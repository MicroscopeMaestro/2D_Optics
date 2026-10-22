import json
with open(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json', 'r') as f:
    d = json.load(f)
surfs = d.get('surfaces', [])
for i in [15, 16, 20]:
    s = surfs[i]
    print(f"Index {i} ({s.get('name')}):")
    for k in ['surface_type', 'glass', 'thickness', 'semi_diameter', 'aperture_type', 'aperture_x_half', 'aperture_y_half', 'substrate_shape', 'tilt_x', 'tilt_y', 'is_air_space', 'is_stop']:
        print(f"  {k}: {s.get(k)}")

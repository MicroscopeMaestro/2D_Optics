import json

with open('spectrometer_raman_50mm_100mm.json') as f:
    config = json.load(f)

surfs = config['surfaces']

# Print all radii and glass for all optical surfaces
print(f"{'Idx':>3} {'Name':35} {'Radius':>10} {'Thickness':>10} {'Glass':>12}")
print('-' * 80)
for i, s in enumerate(surfs):
    r = s.get('radius', 0)
    t = s.get('thickness', 0)
    g = s.get('glass', 'AIR')
    name = s.get('name', '')
    print(f"{i:>3} {name:35} {r:>10.3f} {t:>10.3f} {g:>12}")

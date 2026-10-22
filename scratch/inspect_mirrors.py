import json, glob

for fname in sorted(glob.glob('*mirror*.json') + glob.glob('thorlabs*.json') + glob.glob('*nir*.json')):
    try:
        with open(fname) as f:
            d = json.load(f)
            surfs = [f"{s.get('name')}: R={s.get('radius')}, T={s.get('thickness')}, Glass={s.get('glass')}, Type={s.get('surface_type')}, DecY={s.get('decenter_y')}, TiltY={s.get('tilt_y')}, TiltX={s.get('tilt_x')}, Semi={s.get('semi_diameter')}, Lines={s.get('grating_lines_per_mm')}" for s in d.get('surfaces', [])]
            print(f'=== {fname} ===')
            print('Wavelengths:', d.get('wavelengths'))
            for s in surfs:
                print('  ', s)
            print()
    except Exception as e:
        print(f'Error {fname}: {e}')

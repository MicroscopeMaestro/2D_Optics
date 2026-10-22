import json
import os

path = 'c:/Users/Juan/Desktop/second_mirror_spectrometer_NIR.json'
if os.path.exists(path):
    with open(path, 'r') as f:
        d = json.load(f)
    
    modified = False
    for s in d['surfaces']:
        if 'A230' in s['name'] and s['surface_type'] == 'Aspheric':
            s['radius'] = 3.50
            s['conic'] = 0.390203
            s['asphere_coeffs'] = [-0.0014534147, -0.00086522294, 0.00012701235, -1.409039e-05]
            s['glass'] = 'S-NPH1'
            modified = True
            print(f"Fixed A230-C prescription in surface {s['name']}")
    
    if modified:
        with open(path, 'w') as f:
            json.dump(d, f, indent=4)
        print("Updated second_mirror_spectrometer_NIR.json")
    else:
        print("No A230 found to fix.")

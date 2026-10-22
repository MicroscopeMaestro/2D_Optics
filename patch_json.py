import json
import copy

with open('spectrometer_raman_50mm_100mm_manual.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_surfaces = []
for s in data['surfaces']:
    if s['name'] == 'Grating_600lpmm':
        orig_thick = s['thickness']
        s['thickness'] = 0.0
        new_surfaces.append(s)
        
        dummy = copy.deepcopy(s)
        dummy['name'] = 'Air Gap (Grating)'
        dummy['surface_type'] = 'Standard'
        dummy['glass'] = 'AIR'
        dummy['thickness'] = orig_thick
        dummy['grating_lines_per_mm'] = 0.0
        # Clear specific properties
        dummy['is_active'] = True
        dummy['is_locked'] = False
        new_surfaces.append(dummy)
        
    elif s['name'] == 'DMD4500_Micromirrors':
        orig_thick = s['thickness']
        s['thickness'] = 0.0
        new_surfaces.append(s)
        
        dummy = copy.deepcopy(s)
        dummy['name'] = 'Air Gap (DMD)'
        dummy['surface_type'] = 'Standard'
        dummy['glass'] = 'AIR'
        dummy['thickness'] = orig_thick
        dummy['mechanism_id'] = ''
        new_surfaces.append(dummy)
    else:
        new_surfaces.append(s)

data['surfaces'] = new_surfaces

with open('spectrometer_raman_50mm_100mm_manual.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Modified JSON")

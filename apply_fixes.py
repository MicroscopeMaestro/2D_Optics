
import json
for fname in ['Spectrometer_Hybrid_1064_Repaired.json', 'spectrometer_nir_900_1700nm.json']:
    with open('spectrometer_relay_redesign/' + fname, 'r') as f: data = json.load(f)
    
    m1_idx = next(i for i,s in enumerate(data['surfaces']) if 'M1' in s['name'])
    grat_idx = next(i for i,s in enumerate(data['surfaces']) if 'Grating' in s['name'])
    
    data['surfaces'][m1_idx]['tilt_x'] = -45.0
    data['surfaces'][grat_idx]['tilt_x'] = 15.0
    
    with open('spectrometer_relay_redesign/' + fname, 'w') as f: json.dump(data, f, indent=4)


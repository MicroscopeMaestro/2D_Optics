
import json
with open('mirror_spectrometer/all_mirror_spectrometer.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'M1' in s['name']: m1 = s
    if 'Grating' in s['name']: grat = s
    if 'M2' in s['name']: m2 = s
    if 'DMD' in s['name']: dmd = s
    if 'M3' in s['name']: m3 = s
    if 'M4' in s['name']: m4 = s
    if 'Detector' in s['name']: det = s

m1['tilt_x'] = 20.0

grat['grating_primary_wavelength_um'] = 1.225
grat['tilt_x'] = -28.1

m2['name'] = 'M2_CM254-025-P01_Focusing'
m2['radius'] = -50.8
m2['thickness'] = 25.4
m2['tilt_x'] = 15.0

dmd['thickness'] = 200.0

m3['name'] = 'M3_CM254-200-P01_Relay_Collector'
m3['radius'] = -400.0
m3['thickness'] = 219.0

with open('mirror_spectrometer/Spectrometer_All_Mirror_1064.json', 'w') as f: json.dump(data, f, indent=4)


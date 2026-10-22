import json
with open('mirror_spectrometer/Spectrometer_All_Mirror_With_AirGaps.json', 'r') as f:
    data = json.load(f)
for s in data['surfaces']:
    print(s.get('name'), "is_locked:", s.get('is_locked'))

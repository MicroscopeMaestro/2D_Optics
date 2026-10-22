
import json
data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

for i, s in enumerate(data['surfaces']):
    if '030-C' in s.get('name', ''):
        print(i, s['name'], s.get('radius'), s.get('thickness'), s.get('glass'))


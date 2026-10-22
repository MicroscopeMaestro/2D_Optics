import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('SENSOR_TUNNEL_RADIUS', 'OPTICS_POCKET_RADIUS')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)

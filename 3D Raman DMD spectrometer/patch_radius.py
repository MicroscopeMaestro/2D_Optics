import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('OPTICS_POCKET_RADIUS        = 12.7', 'OPTICS_POCKET_RADIUS        = 12.90')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated OPTICS_POCKET_RADIUS to 12.90")

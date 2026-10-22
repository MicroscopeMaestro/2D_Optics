import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('LENS_EXTRA_THICKNESS_30_100 = 0.5', 'LENS_EXTRA_THICKNESS_30_100 = 1.0')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Extra thickness updated!")

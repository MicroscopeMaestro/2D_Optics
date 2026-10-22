import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('DRILL_APERTURE_DEPTH', 'DRILL_EXIT_LENGTH')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)

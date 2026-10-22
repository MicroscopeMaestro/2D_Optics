import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('ENABLE_STRUCTURAL_RIBBING   = True', 'ENABLE_STRUCTURAL_RIBBING   = False')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Disabled structural ribbing!")

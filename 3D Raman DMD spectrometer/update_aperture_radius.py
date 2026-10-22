with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update default aperture_radius from 10.00 to 10.50
content = re.sub(r'aperture_radius = 10\.00', 'aperture_radius = 10.50', content)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated build_external_bed.py with aperture_radius = 10.50!")

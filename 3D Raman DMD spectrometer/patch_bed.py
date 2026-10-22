with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Change pocket_radius from 13.4 to 13.0
content = re.sub(r'pocket_radius = 13\.4', 'pocket_radius = 13.0', content)

# Change the default output step name to the new version
content = re.sub(r'output_step="external_aspheric_bed\.step"', 'output_step="external_aspheric_bed_13mm.step"', content)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated build_external_bed.py for 13mm pocket radius!")

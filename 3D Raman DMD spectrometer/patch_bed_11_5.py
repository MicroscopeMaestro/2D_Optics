with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Change pocket_radius back to 13.4
content = re.sub(r'pocket_radius = 13\.0', 'pocket_radius = 13.4', content)

# Modify r_cleared logic to step down to 11.5 instead of ~12.15
new_r_logic = '''        if r > 12.0:
            r_cleared = p_radius
        else:
            r_cleared = min(11.5, max(r + c_clearance, 10.0))'''

content = re.sub(r'r_cleared = p_radius if r > 12\.0 else max\(r \+ c_clearance, 10\.0\)', new_r_logic, content)

# Output to a new step file
content = re.sub(r'output_step="external_aspheric_bed_13mm\.step"', 'output_step="external_aspheric_bed_11_5mm.step"', content)
content = re.sub(r'output_step="external_aspheric_bed\.step"', 'output_step="external_aspheric_bed_11_5mm.step"', content)

with open('build_external_bed_11_5.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Created script build_external_bed_11_5.py for the 11.5mm shelf version!")

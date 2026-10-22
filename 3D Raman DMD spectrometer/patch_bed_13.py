with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Keep pocket_radius at 13.4
content = re.sub(r'pocket_radius = 13\.0', 'pocket_radius = 13.4', content)

# Modify r_cleared logic
new_r_logic = '''        if r > 12.0:
            r_cleared = p_radius
        else:
            r_cleared = max(r + 1.0, 10.0)'''

content = re.sub(r'r_cleared = p_radius if r > 12\.0 else max\(r \+ c_clearance, 10\.0\)', new_r_logic, content)

# Output to a new step file
content = re.sub(r'output_step="external_aspheric_bed\.step"', 'output_step="external_aspheric_bed_13_0mm_shelf.step"', content)

with open('build_external_bed_13_0.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Created script build_external_bed_13_0.py!")

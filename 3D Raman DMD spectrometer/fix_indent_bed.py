with open('build_external_bed_11_5.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'if r > 12.0:' in line:
        lines[i] = '        if r > 12.0:\n'
    elif 'r_cleared = p_radius' in line and 'else:' not in lines[i+1]:
        lines[i] = '            r_cleared = p_radius\n'
    elif 'else:' in line and 'if r > 12.0:' in lines[i-2]:
        lines[i] = '        else:\n'
    elif 'r_cleared = min(11.5' in line:
        lines[i] = '            r_cleared = min(11.5, max(r + c_clearance, 10.0))\n'

with open('build_external_bed_11_5.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Fixed indent in build_external_bed_11_5.py!")

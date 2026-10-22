import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the r_cleared logic
pattern = r'r_cleared = min\(max\(r \+ clearance, 9\.0\), pocket_radius\).*?\n'
replacement = '''if r > 12.0:
            r_cleared = pocket_radius
        else:
            r_cleared = min(max(r + 0.25, 9.0), pocket_radius)
'''

content = re.sub(pattern, replacement, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated make_aspheric_cradle clearance logic!")

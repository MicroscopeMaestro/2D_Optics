import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r"extra_z = lens_extra_thickness_30_100 if any\(k in low for k in \['030', '100'\]\) else 0\.0"
replacement = '''extra_z = lens_extra_thickness_30_100 if any(k in low for k in ['030', '100']) else 0.0
            if "filter" in low:
                extra_z = 0.5'''

content = re.sub(pattern, replacement, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Edge Filter extra_z!")

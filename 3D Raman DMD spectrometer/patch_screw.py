with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_screw = "(-90.0, 115.0),  # LEFT MID 1 (moved outward)"
new_screw = "(-82.0, 115.0),  # LEFT MID 1 (moved inward to match wall)"
content = content.replace(old_screw, new_screw)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Left Mid 1 screw!")

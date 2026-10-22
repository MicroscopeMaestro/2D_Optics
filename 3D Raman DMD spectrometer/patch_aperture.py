with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = "r_cleared = min(max(r + clearance, 9.0), pocket_radius)"
new_logic = "r_cleared = min(max(r + clearance, 9.5), pocket_radius)"
content = content.replace(old_logic, new_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Increased aspheric clear aperture radius to 9.5mm!")

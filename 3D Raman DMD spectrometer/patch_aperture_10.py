with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("r_cleared = min(max(r + clearance, 9.5), pocket_radius)", "r_cleared = min(max(r + clearance, 10.0), pocket_radius)")
content = content.replace("preserving the 19mm clear aperture (minimum radius 9.5mm)", "preserving the 20mm clear aperture (minimum radius 10.0mm)")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Increased aspheric clear aperture radius to 10.0mm (20mm diameter)!")

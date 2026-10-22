with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Increase pocket width (X) from 36.8 to 37.4 (0.7mm gap on both sides)
content = content.replace('p_width = 36.8', 'p_width = 37.4')

# Increase pocket height (Y) from 32.8 to 33.4 (0.7mm gap on both sides)
content = content.replace('p_height = 32.8', 'p_height = 33.4')

# Shift the starting coordinate to maintain symmetry around the block
content = content.replace('FreeCAD.Vector(-14.4, -16.4, p_z_min)', 'FreeCAD.Vector(-14.7, -16.7, p_z_min)')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer to have MASSIVE 0.7mm slip-fit tolerances for the cartridge!")

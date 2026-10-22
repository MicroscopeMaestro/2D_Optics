with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# p_height is the Y-axis width (was 28.8, now 32.8)
content = content.replace('p_height = 28.8', 'p_height = 32.8')
content = content.replace('FreeCAD.Vector(-14.4, -14.4, p_z_min)', 'FreeCAD.Vector(-14.4, -16.4, p_z_min)')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer script pockets for 32.8mm Y-width!")

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('p_height = 28.8', 'p_height = 36.8')
content = content.replace('box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -p_height/2, p_z_min))',
                          'box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -14.4, p_z_min))')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer pocket to match rectangular profile!")

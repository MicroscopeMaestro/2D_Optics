with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_gr_box = "gr_box = Part.makeBox(26.0, 26.0, 6.5, FreeCAD.Vector(-13.0, -13.0, -3.25))"
new_gr_box = "gr_box = Part.makeBox(26.0, 26.0, 6.2, FreeCAD.Vector(-13.0, -13.0, -3.1))"
content = content.replace(old_gr_box, new_gr_box)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated grating pocket thickness to 6.2mm!")

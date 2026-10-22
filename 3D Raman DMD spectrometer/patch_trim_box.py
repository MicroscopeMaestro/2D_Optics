with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_trim = '''    # Clean left boundary at Y = -80.0
    left_trim_box = Part.makeBox(200, 100, 300, FreeCAD.Vector(-100, -180.0, -50))'''
new_trim = '''    # Clean left boundary at Y = -120.0
    left_trim_box = Part.makeBox(200, 100, 300, FreeCAD.Vector(-100, -220.0, -50))'''
content = content.replace(old_trim, new_trim)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Pushed back the rogue left_trim_box!")

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_cc2 = '''    # 2. Left entrance corner at Y=-22, Z=50
    cc2 = make_chamfer_cut(-22, 45, -27, 53, -10, 40, 60)'''
new_cc2 = '''    # 2. (Removed) Left entrance corner at Y=-22, Z=50
    cc2 = Part.makeBox(1,1,1, FreeCAD.Vector(0,0,-1000)) # dummy'''
content = content.replace(old_cc2, new_cc2)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed cc2 chamfer cut!")

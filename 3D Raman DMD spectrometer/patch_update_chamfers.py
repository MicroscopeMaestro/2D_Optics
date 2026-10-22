with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_cc3 = '''    # 3. Left DMD bottom corner at Y=-110, Z=95
    cc3 = make_chamfer_cut(-105, 92, -110, 100, -125, 85, 105)'''
new_cc3 = '''    # 3. Left DMD bottom corner at Y=-86, Z=95
    cc3 = make_chamfer_cut(-82, 92, -86, 100, -100, 85, 105)'''
content = content.replace(old_cc3, new_cc3)

old_cc4 = '''    # 4. Left DMD top corner at Y=-110, Z=132
    cc4 = make_chamfer_cut(-110, 127, -105, 137, -125, 120, 145)'''
new_cc4 = '''    # 4. Left DMD top corner at Y=-86, Z=120
    cc4 = make_chamfer_cut(-86, 115, -82, 125, -100, 110, 135)'''
content = content.replace(old_cc4, new_cc4)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated corner chamfers!")

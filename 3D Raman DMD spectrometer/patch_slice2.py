import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

slice_code = '''
    # 8. Slicing off the front initial airgap entirely so the 1st aspheric lens tip is strictly at the surface
    # The casing front is sliced off precisely at Z=15.36 (the origin of the first lens)
    front_slice_box = Part.makeBox(200, 200, 35.36, FreeCAD.Vector(-100, -100, -20.0))
    top_half, bottom_half = top_half.cut(front_slice_box), bottom_half.cut(front_slice_box)
'''

content = content.replace('    top_half = clean_solids(top_half)', slice_code + '\n    top_half = clean_solids(top_half)')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Changes applied!")

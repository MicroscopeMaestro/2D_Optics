import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('bb_flange_face = Part.Face(slice_shapes[0]).makeOffset2D(20.0)', 'bb_flange_face = Part.Face(slice_shapes[0]).makeOffset2D(12.0)')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated flange to 12.0")

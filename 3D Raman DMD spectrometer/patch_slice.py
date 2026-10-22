import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

slice_code = '''
    # Slice the front of the casing so the first aspheric lens tip is perfectly flush with the edge of the surface
    # The casing initially starts at Z=0, but the first lens is at Z=15.36
    front_slice_box = Part.makeBox(200, 200, 35.36, FreeCAD.Vector(-100, -100, -20.0))
    top_half, bottom_half = top_half.cut(front_slice_box), bottom_half.cut(front_slice_box)
'''

# Insert after the taper loop
pattern = r'(for taper in \[.*?\]:\n\s+top_half, bottom_half = top_half\.cut\(taper\), bottom_half\.cut\(taper\))'
content = re.sub(pattern, r'\1\n' + slice_code, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added front slice box!")

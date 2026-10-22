
with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace('lines_per_um=0.150', 'grating_lines_per_mm=150.0')
content = content.replace('diffraction_order=1', 'grating_order=1')
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)


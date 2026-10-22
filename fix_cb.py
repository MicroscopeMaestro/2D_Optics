
with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace('SurfaceType.COORD_BREAK', 'SurfaceType.COORDINATE_BREAK')
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)


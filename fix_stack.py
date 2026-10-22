
with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace('stack = SurfaceStack(surfaces)', 'stack = SurfaceStack(); stack.surfaces = surfaces')
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)


with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace("[h['pos'] for h in r.history]", "[[h['x_global'], h['y_global'], h['z_global']] for h in r.history]")
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)

with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace('sys_opt.stack.config.wavelengths = [0.8, 1.75, 2.7]', 'sys_opt.source_config.wavelengths = "0.8, 1.75, 2.7"')
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)

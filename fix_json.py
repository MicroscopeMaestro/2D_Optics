
with open('build_nir_mirror.py', 'r') as f:
    content = f.read()
content = content.replace('sys_opt.to_json', 'sys_opt.stack.save_to_json')
with open('build_nir_mirror.py', 'w') as f:
    f.write(content)


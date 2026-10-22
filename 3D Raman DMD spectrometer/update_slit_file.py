with open('run_rotate_slits.py', 'r', encoding='utf-8') as f:
    code = f.read()

with open('build_slit_cartridge.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("build_slit_cartridge.py updated with 90-degree rotated slit!")

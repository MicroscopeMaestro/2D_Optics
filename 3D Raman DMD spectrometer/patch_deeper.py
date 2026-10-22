with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('z_max = bb.ZMax + 1.0', 'z_max = bb.ZMax + 8.0')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to be deeper!")

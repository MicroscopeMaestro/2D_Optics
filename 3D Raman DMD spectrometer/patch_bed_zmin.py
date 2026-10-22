with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Extend ZMin by 8.0mm
content = content.replace('z_min = bb.ZMin - 0.1', 'z_min = bb.ZMin - 8.0')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed ZMin!")

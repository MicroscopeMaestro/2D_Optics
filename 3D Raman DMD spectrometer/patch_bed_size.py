with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('width = 30.0', 'width = 28.0')
content = content.replace('height = 30.0', 'height = 28.0')
content = content.replace('z_max = bb.ZMax + 5.0', 'z_max = bb.ZMax + 1.0')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to 28x28mm and thinner depth!")

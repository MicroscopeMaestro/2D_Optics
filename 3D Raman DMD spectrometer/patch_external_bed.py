with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('width = 40.0', 'width = 30.0')
content = content.replace('height = 40.0', 'height = 30.0')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated external bed script to 30x30mm!")

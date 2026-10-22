with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Increase from 13.3 to 13.4 (makes the diameter 26.8mm for a 25.4mm lens)
content = content.replace('pocket_radius = 13.3', 'pocket_radius = 13.4')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to have looser lens tolerance!")

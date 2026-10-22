path = 'optics_app/engine/tracer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('semi_diameter\', 25.0', 'semi_diameter\', 12.7')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

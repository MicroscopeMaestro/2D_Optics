import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    prefix = match.group(1)
    obj = match.group(2)
    return f"{prefix}if hasattr({obj}, 'set_picker'): {obj}.set_picker(5); {obj}._surface_index = surf_idx\n{prefix}self.ax.add_collection3d({obj})"

# Pattern matches: <whitespace>self.ax.add_collection3d(<var_name>)
pattern = r"(\s+)self\.ax\.add_collection3d\(([a-zA-Z0-9_]+)\)"
content = re.sub(pattern, replacer, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

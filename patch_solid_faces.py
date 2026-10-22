import re

with open('optics_app/gui/visualizer_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''                self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                self.ax.plot(rear_ring[2], rear_ring[0], rear_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                solid_face_ids.update((id(front), id(rear)))'''

replacement = '''                self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                self.ax.plot(rear_ring[2], rear_ring[0], rear_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                if is_glass_region:
                    solid_face_ids.update((id(front), id(rear)))'''

if target in content:
    content = content.replace(target, replacement)
    with open('optics_app/gui/visualizer_3d.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched solid_face_ids")
else:
    print("Could not find target")

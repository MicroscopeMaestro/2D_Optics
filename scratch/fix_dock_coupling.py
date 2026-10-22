import re

path = 'optics_app/gui/component_editor_dock.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix _on_aperture_x_changed
old_x = '''        if surf:
            surf.aperture_x_half = float(val)
            hx = surf.aperture_x_half
            hy = surf.aperture_y_half if surf.aperture_y_half > 0 else float(surf.semi_diameter)
            surf.semi_diameter = float(np.sqrt(hx**2 + hy**2))
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")'''

new_x = '''        if surf:
            surf.aperture_x_half = float(val)
            hx = surf.aperture_x_half
            hy = surf.aperture_y_half
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")'''

content = content.replace(old_x, new_x)

# Fix _on_aperture_y_changed
old_y = '''        if surf:
            surf.aperture_y_half = float(val)
            hx = surf.aperture_x_half if surf.aperture_x_half > 0 else float(surf.semi_diameter)
            hy = surf.aperture_y_half
            surf.semi_diameter = float(np.sqrt(hx**2 + hy**2))
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")'''

new_y = '''        if surf:
            surf.aperture_y_half = float(val)
            hx = surf.aperture_x_half
            hy = surf.aperture_y_half
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")'''

content = content.replace(old_y, new_y)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

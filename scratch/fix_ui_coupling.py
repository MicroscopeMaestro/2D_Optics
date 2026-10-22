import re

path = 'optics_app/gui/component_editor_dock.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''            if type_name == "Point":
                surf.semi_diameter = 0.0
                surf.aperture_x_half = 0.0
                surf.aperture_y_half = 0.0
                if hasattr(self, 'widget_rect_aperture'):
                    self.widget_rect_aperture.setVisible(False)
                if hasattr(self, 'lbl_aperture_readout'):
                    self.lbl_aperture_readout.setText("Point Object / Image: ∅ = 0.00 mm (Zero Extent)")'''

new_logic = '''            if type_name == "Point":
                surf.aperture_x_half = 0.0
                surf.aperture_y_half = 0.0
                if hasattr(self, 'widget_rect_aperture'):
                    self.widget_rect_aperture.setVisible(False)
                if hasattr(self, 'lbl_aperture_readout'):
                    self.lbl_aperture_readout.setText(f"Point Object / Image | Semi-D: {surf.semi_diameter:.2f} mm")'''

content = content.replace(old_logic, new_logic)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

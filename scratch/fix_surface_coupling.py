import re

path = 'optics_app/engine/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_circ = '''        elif atype == "Circular":
            if "R" in kv:
                self.semi_diameter = float(kv["R"])
                self.aperture_r_outer = float(kv["R"])
            elif "ROUT" in kv:
                self.semi_diameter = float(kv["ROUT"])
                self.aperture_r_outer = float(kv["ROUT"])'''

new_circ = '''        elif atype == "Circular":
            if "R" in kv:
                self.aperture_r_outer = float(kv["R"])
            elif "ROUT" in kv:
                self.aperture_r_outer = float(kv["ROUT"])'''

old_rect = '''            if hx > 0 and hy > 0:
                self.semi_diameter = float(np.sqrt(hx**2 + hy**2))
            elif hx > 0:
                self.semi_diameter = hx
            elif hy > 0:
                self.semi_diameter = hy'''

new_rect = '''            # Removed coupling: semi_diameter remains independent'''

old_ell = '''            self.semi_diameter = max(self.aperture_a_semi, self.aperture_b_semi)'''
new_ell = '''            # Removed coupling: semi_diameter remains independent'''

old_ann = '''                self.semi_diameter = self.aperture_r_outer'''
new_ann = '''                # Removed coupling: semi_diameter remains independent'''

content = content.replace(old_circ, new_circ)
content = content.replace(old_rect, new_rect)
content = content.replace(old_ell, new_ell)
content = content.replace(old_ann, new_ann)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

import re

path = 'optics_app/gui/lde_table.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic_1 = '''                if i != row and (s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False)):
                    s.surface_type = SurfaceType.STANDARD
                    s.is_stop = False
                    s.is_aperture_stop = False'''

new_logic_1 = '''                if i != row and (s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False)):
                    if s.surface_type == SurfaceType.STOP:
                        s.surface_type = SurfaceType.STANDARD
                    s.is_stop = False
                    s.is_aperture_stop = False'''

old_logic_2 = '''            target_surf.surface_type = SurfaceType.STOP
            target_surf.is_stop = True'''

new_logic_2 = '''            target_surf.is_stop = True'''

old_logic_3 = '''        else:
            target_surf = stack.surfaces[row]
            target_surf.surface_type = SurfaceType.STANDARD
            target_surf.is_stop = False'''

new_logic_3 = '''        else:
            target_surf = stack.surfaces[row]
            if target_surf.surface_type == SurfaceType.STOP:
                target_surf.surface_type = SurfaceType.STANDARD
            target_surf.is_stop = False'''

old_logic_4 = '''        for s in stack.surfaces:
            if s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False):
                s.surface_type = SurfaceType.STANDARD
                s.is_stop = False'''

new_logic_4 = '''        for s in stack.surfaces:
            if s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False):
                if s.surface_type == SurfaceType.STOP:
                    s.surface_type = SurfaceType.STANDARD
                s.is_stop = False'''

content = content.replace(old_logic_1, new_logic_1)
content = content.replace(old_logic_2, new_logic_2)
content = content.replace(old_logic_3, new_logic_3)
content = content.replace(old_logic_4, new_logic_4)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

import re

path = 'optics_app/engine/lde.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        for d in surfaces_data:
            st = d.get("surface_type")
            if isinstance(st, str):
                for val in [SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.MIRROR, SurfaceType.ASPHERIC, SurfaceType.STOP, SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.CAD, SurfaceType.AIR_SPACE]:
                    if val.upper() == st.upper():
                        d["surface_type"] = val
                        break'''

new_logic = '''        for d in surfaces_data:
            st = d.get("surface_type")
            
            # Zemax structure migration: Stop is no longer a type, it's a flag
            if isinstance(st, str) and st.upper() == "STOP":
                d["surface_type"] = SurfaceType.STANDARD
                d["is_stop"] = True
                st = SurfaceType.STANDARD
                
            if isinstance(st, str):
                for val in [SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.MIRROR, SurfaceType.ASPHERIC, SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.CAD, SurfaceType.AIR_SPACE]:
                    if val.upper() == st.upper():
                        d["surface_type"] = val
                        break'''

content = content.replace(old_logic, new_logic)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

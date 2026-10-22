import re

with open('optics_app/cad/exporter.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        # 7. Mirror, Grating, DMD, Detector'''

replacement = '''        # 7. Air Gap
        elif surf.thickness > 0 and i < len(stack.surfaces) - 1 and surf.glass.upper() in ["AIR", "", "NONE"] and not getattr(surf, "is_air_space", False):
            if 0 < float(surf.thickness) < 1000.0:
                s1 = surf
                s2 = stack.surfaces[i + 1]
                name = surf.name or f"Air_Gap_{elem_idx + 1}"
                if not name or name == f"S{i}":
                    name = f"Air Gap (S{i}-S{i+1})"
                max_sd = max(float(s1.semi_diameter), float(s2.semi_diameter))
                elements.append({
                    "index": elem_idx,
                    "name": name,
                    "type": "Air Gap",
                    "surface_indices": [i, i + 1],
                    "surfaces": [s1, s2],
                    "material": "Air",
                    "semi_diameter": max_sd,
                    "clear_aperture_diam": 2.0 * max_sd,
                    "z_position": float(s1.global_origin[2]),
                    "origin": s1.global_origin,
                    "is_default_selected": False,
                })
                elem_idx += 1
            i += 1
            continue

        # 8. Mirror, Grating, DMD, Detector'''

content = content.replace(target, replacement)

with open('optics_app/cad/exporter.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("done")

"""
Optical Component & Lens Element Color Styling System.
Provides deterministic color assignment, glass tinting, element group identification,
and visual theme palettes for LDE tables, 2D ray diagrams, and 3D visualizers.
"""

from typing import List, Dict, Tuple, Any, Optional

# Curated optical element color palette (background tint, border/fill color, text/badge color)
LENS_ELEMENT_PALETTES = [
    {
        "name": "Cyan Blue",
        "bg_hex": "#E1F5FE",      # Light sky blue
        "fill_hex": "#4FC3F7",    # 2D glass fill
        "border_hex": "#0288D1",  # Lens boundary line
        "badge_hex": "#01579B",   # Badge text
        "rgb": (0.31, 0.76, 0.97),
    },
    {
        "name": "Emerald Green",
        "bg_hex": "#E8F5E9",      # Light mint green
        "fill_hex": "#81C784",    # 2D glass fill
        "border_hex": "#2E7D32",  # Lens boundary line
        "badge_hex": "#1B5E20",   # Badge text
        "rgb": (0.51, 0.78, 0.52),
    },
    {
        "name": "Coral Amber",
        "bg_hex": "#FFF3E0",      # Light peach
        "fill_hex": "#FFB74D",    # 2D glass fill
        "border_hex": "#E65100",  # Lens boundary line
        "badge_hex": "#BF360C",   # Badge text
        "rgb": (1.0, 0.72, 0.30),
    },
    {
        "name": "Lavender Purple",
        "bg_hex": "#EDE7F6",      # Light lavender
        "fill_hex": "#B39DDB",    # 2D glass fill
        "border_hex": "#512DA8",  # Lens boundary line
        "badge_hex": "#311B92",   # Badge text
        "rgb": (0.70, 0.62, 0.86),
    },
    {
        "name": "Aquamarine Teal",
        "bg_hex": "#E0F2F1",      # Light teal
        "fill_hex": "#4DB6AC",    # 2D glass fill
        "border_hex": "#00796B",  # Lens boundary line
        "badge_hex": "#004D40",   # Badge text
        "rgb": (0.30, 0.71, 0.67),
    },
    {
        "name": "Rose Pink",
        "bg_hex": "#FCE4EC",      # Light rose pink
        "fill_hex": "#F06292",    # 2D glass fill
        "border_hex": "#C2185B",  # Lens boundary line
        "badge_hex": "#880E4F",   # Badge text
        "rgb": (0.94, 0.38, 0.57),
    },
    {
        "name": "Golden Sand",
        "bg_hex": "#FFFDE7",      # Pale yellow
        "fill_hex": "#FFF176",    # 2D glass fill
        "border_hex": "#FBC02D",  # Lens boundary line
        "badge_hex": "#F57F17",   # Badge text
        "rgb": (1.0, 0.95, 0.46),
    },
    {
        "name": "Royal Indigo",
        "bg_hex": "#E8EAF6",      # Soft indigo
        "fill_hex": "#7986CB",    # 2D glass fill
        "border_hex": "#303F9F",  # Lens boundary line
        "badge_hex": "#1A237E",   # Badge text
        "rgb": (0.47, 0.53, 0.80),
    }
]

SPECIAL_SURFACE_PALETTES = {
    "STOP": {
        "name": "Stop",
        "bg_hex": "#F5F5F5",      # Neutral soft gray
        "fill_hex": "#CCCCCC",
        "border_hex": "#9E9E9E",
        "badge_hex": "#616161",
        "rgb": (0.75, 0.75, 0.75),
    },
    "DETECTOR": {
        "name": "Detector / Image",
        "bg_hex": "#F5F5F5",      # Neutral soft gray
        "fill_hex": "#CCCCCC",
        "border_hex": "#9E9E9E",
        "badge_hex": "#616161",
        "rgb": (0.75, 0.75, 0.75),
    },
    "MIRROR": {
        "name": "Mirror",
        "bg_hex": "#FFF8E1",      # Soft gold
        "fill_hex": "#FFE082",
        "border_hex": "#FF8F00",
        "badge_hex": "#E65100",
        "rgb": (1.0, 0.88, 0.51),
    },
    "GRATING": {
        "name": "Grating",
        "bg_hex": "#EDE7F6",      # Soft violet
        "fill_hex": "#BA68C8",
        "border_hex": "#8E24AA",
        "badge_hex": "#4A148C",
        "rgb": (0.73, 0.41, 0.78),
    },
    "OBJECT": {
        "name": "Object",
        "bg_hex": "#F5F5F5",      # Neutral soft gray
        "fill_hex": "#CCCCCC",
        "border_hex": "#9E9E9E",
        "badge_hex": "#616161",
        "rgb": (0.75, 0.75, 0.75),
    }
}


import re

def extract_base_group_name(name: str) -> str:
    """Extract common base group name from surface name."""
    if not name:
        return ""
    # Strip common suffixes like _S1, _S2, _Front, _Mid, _Back, _Rear, _Surface1, _Elem1, etc.
    cleaned = re.sub(r'(_|\s+)(S\d+|Front|Mid|Middle|Back|Rear|Surface\d+|Surf\d+|Elem\d+|Element\d+|Lens\d+)$', '', name.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def is_object_plane(surf: Any, index: int) -> bool:
    """Check if surface is the object reference plane (not an Objective lens)."""
    stype = str(getattr(surf, 'surface_type', '')).upper()
    sname = str(getattr(surf, 'name', '')).upper().strip()
    if "OBJECTIVE" in sname:
        return False
    if index == 0:
        if "OBJECT" in stype or sname in ("OBJ", "OBJECT", "S0 (OBJ)", "S0", "S0(OBJ)") or "(OBJ)" in sname or not sname:
            return True
    if "OBJECT" in stype and sname in ("OBJ", "OBJECT", "S0 (OBJ)", "S0", "S0(OBJ)"):
        return True
    return False


def is_stop_plane(surf: Any) -> bool:
    """Check if surface is an aperture stop."""
    stype = str(getattr(surf, 'surface_type', '')).upper()
    sname = str(getattr(surf, 'name', '')).upper().strip()
    if getattr(surf, 'is_stop', False) or getattr(surf, 'is_aperture_stop', False):
        return True
    if "STOP" in stype:
        return True
    if sname in ("STO", "STOP", "APERTURE STOP", "APER STOP") or "(STO)" in sname or "(STOP)" in sname:
        return True
    return False


def is_detector_plane(surf: Any, index: int, total_surfaces: int) -> bool:
    """Check if surface is the detector / image plane."""
    stype = str(getattr(surf, 'surface_type', '')).upper()
    sname = str(getattr(surf, 'name', '')).upper().strip()
    if index == total_surfaces - 1:
        if "DETECTOR" in stype or "IMAGE" in stype or sname in ("IMA", "IMJ", "IMAGE", "DETECTOR", f"S{index} (IMA)", f"S{index} (IMJ)") or "(IMA)" in sname or "(IMJ)" in sname or "(IMAGE)" in sname:
            return True
    if "DETECTOR" in stype or sname in ("IMA", "IMJ", "IMAGE", "DETECTOR") or "(IMA)" in sname or "(IMJ)" in sname:
        return True
    return False


def compute_surface_groups(surfaces: List[Any]) -> List[Dict[str, Any]]:
    """
    Computes deterministic element group assignments for all surfaces in a stack.
    Handles:
    - Explicit element_id or group_name attributes
    - Common name prefixes (e.g. 'Objective_S1', 'Objective_S2' -> 'Objective')
    - Consecutive solid glass lenses with cemented interfaces and back air boundary
    - Neutral styling for OBJ, STOP, and IMJ/Detector
    """
    if not surfaces:
        return []

    n = len(surfaces)
    results = [None] * n

    # 1. First pass: Identify special non-lens reference surfaces
    for i, surf in enumerate(surfaces):
        stype = str(getattr(surf, 'surface_type', '')).upper()
        glass = str(getattr(surf, 'glass', '')).upper().strip()

        if is_object_plane(surf, i):
            results[i] = {"group_type": "OBJECT", "element_id": -1, "group_name": "Object", "palette": SPECIAL_SURFACE_PALETTES["OBJECT"]}
        elif is_detector_plane(surf, i, n):
            results[i] = {"group_type": "DETECTOR", "element_id": -1, "group_name": "Detector", "palette": SPECIAL_SURFACE_PALETTES["DETECTOR"]}
        elif is_stop_plane(surf):
            results[i] = {"group_type": "STOP", "element_id": -1, "group_name": "Stop", "palette": SPECIAL_SURFACE_PALETTES["STOP"]}
        elif "GRATING" in stype or getattr(surf, 'grating_lines_per_mm', 0) > 0:
            results[i] = {"group_type": "GRATING", "element_id": 997, "group_name": "Grating", "palette": SPECIAL_SURFACE_PALETTES["GRATING"]}
        elif "MIRROR" in glass or "MIRROR" in stype or (hasattr(surf, 'is_reflective') and surf.is_reflective()):
            results[i] = {"group_type": "MIRROR", "element_id": 996, "group_name": "Mirror", "palette": SPECIAL_SURFACE_PALETTES["MIRROR"]}

    # 2. Second pass: Collect any explicitly assigned element_ids or group_names
    known_group_to_id = {}
    next_element_id = 0

    for i, surf in enumerate(surfaces):
        if results[i] is not None:
            continue
        e_id = getattr(surf, 'element_id', None)
        g_name = getattr(surf, 'group_name', '') or extract_base_group_name(getattr(surf, 'name', ''))
        if e_id is not None and e_id >= 0:
            if g_name and g_name not in known_group_to_id:
                known_group_to_id[g_name] = e_id
            if e_id >= next_element_id:
                next_element_id = e_id + 1

    # 3. Third pass: Assign groups to remaining unassigned lens surfaces
    i = 0
    while i < n:
        if results[i] is not None:
            i += 1
            continue

        surf = surfaces[i]
        s_glass = str(getattr(surf, 'glass', '')).upper().strip()
        e_id = getattr(surf, 'element_id', None)
        g_name = getattr(surf, 'group_name', '') or extract_base_group_name(getattr(surf, 'name', ''))

        # If explicit or known prefix exists
        if e_id is not None and e_id >= 0:
            elem_id = e_id
            disp_name = g_name or f"Lens {elem_id + 1}"
            pal = LENS_ELEMENT_PALETTES[elem_id % len(LENS_ELEMENT_PALETTES)]
            results[i] = {"group_type": "LENS", "element_id": elem_id, "group_name": disp_name, "palette": pal}
            i += 1
            continue
        elif g_name and g_name in known_group_to_id:
            elem_id = known_group_to_id[g_name]
            disp_name = g_name
            pal = LENS_ELEMENT_PALETTES[elem_id % len(LENS_ELEMENT_PALETTES)]
            results[i] = {"group_type": "LENS", "element_id": elem_id, "group_name": disp_name, "palette": pal}
            i += 1
            continue
        else:
            # Check if consecutive surfaces share the same base name prefix
            prefix_match_span = [i]
            if g_name and len(g_name) > 2 and not g_name.startswith("S"):
                j = i + 1
                while j < n and results[j] is None:
                    next_g_name = getattr(surfaces[j], 'group_name', '') or extract_base_group_name(getattr(surfaces[j], 'name', ''))
                    if next_g_name.lower() == g_name.lower():
                        prefix_match_span.append(j)
                        j += 1
                    else:
                        break

            if len(prefix_match_span) > 1:
                elem_id = next_element_id
                next_element_id += 1
                if g_name:
                    known_group_to_id[g_name] = elem_id
                disp_name = g_name or f"Lens {elem_id + 1}"
                pal = LENS_ELEMENT_PALETTES[elem_id % len(LENS_ELEMENT_PALETTES)]
                for idx_m in prefix_match_span:
                    results[idx_m] = {
                        "group_type": "LENS",
                        "element_id": elem_id,
                        "group_name": disp_name,
                        "palette": pal
                    }
                i = prefix_match_span[-1] + 1
                continue

            # Standard solid glass element: starts at glass, spans to rear air boundary
            is_solid = s_glass and s_glass not in ["AIR", "VACUUM", "MIRROR", "MIRR", "REFL", ""]
            if is_solid:
                elem_id = next_element_id
                next_element_id += 1
                disp_name = g_name or f"Lens {elem_id + 1}"
                if g_name:
                    known_group_to_id[g_name] = elem_id
                pal = LENS_ELEMENT_PALETTES[elem_id % len(LENS_ELEMENT_PALETTES)]

                lens_indices = [i]
                k = i + 1
                while k < n and results[k] is None:
                    lens_indices.append(k)
                    k_glass = str(getattr(surfaces[k], 'glass', '')).upper().strip()
                    if not k_glass or k_glass in ["AIR", "VACUUM"]:
                        break
                    k += 1

                for idx_l in lens_indices:
                    results[idx_l] = {
                        "group_type": "LENS",
                        "element_id": elem_id,
                        "group_name": disp_name,
                        "palette": pal
                    }
                i = lens_indices[-1] + 1
                continue
            else:
                # Standalone air gap
                results[i] = {
                    "group_type": "AIR",
                    "element_id": -1,
                    "group_name": "Air Gap",
                    "palette": SPECIAL_SURFACE_PALETTES["OBJECT"]
                }
                i += 1
                continue

    return results


def get_surface_element_group(surfaces: List[Any], index: int) -> Dict[str, Any]:
    """
    Given a list of surfaces and a surface index, identifies its element group,
    lens ID, role, and returns its assigned color palette.
    """
    if not surfaces or not (0 <= index < len(surfaces)):
        return {"group_type": "AIR", "element_id": -1, "group_name": "Air", "palette": SPECIAL_SURFACE_PALETTES["OBJECT"]}

    groups = compute_surface_groups(surfaces)
    if 0 <= index < len(groups) and groups[index] is not None:
        return groups[index]

    return {"group_type": "AIR", "element_id": -1, "group_name": "Air", "palette": SPECIAL_SURFACE_PALETTES["OBJECT"]}


def get_lens_elements(surfaces: List[Any], unlocked_groups: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Extract all distinct multi-surface or single-surface solid lens element assemblies.
    Returns a list of dictionary descriptors for each lens element.
    """
    if not surfaces:
        return []

    groups = compute_surface_groups(surfaces)
    unlocked = unlocked_groups or set()
    
    # A catalog/group name is a part identity, not an installed-instance
    # identity. Two AC254-050-C lenses must therefore remain two selectable
    # assemblies. Build entries from contiguous physical spans instead of
    # merging every surface that shares element_id or group_name.
    elements: List[Dict[str, Any]] = []
    seen_indices = set()
    for i, grp in enumerate(groups):
        if i in seen_indices or not grp or grp.get("group_type") != "LENS":
            continue

        if getattr(surfaces[i], "is_glass", False):
            glass_seed = i
        elif i > 0 and getattr(surfaces[i - 1], "is_glass", False):
            glass_seed = i - 1
        else:
            # Explicit non-glass lens-like groups are kept contiguous, but are
            # never coalesced with a later installed copy of the same part.
            group_name = str(grp.get("group_name", ""))
            span = [i]
            j = i + 1
            while j < len(groups):
                next_group = groups[j]
                if not next_group or next_group.get("group_type") != "LENS":
                    break
                if str(next_group.get("group_name", "")) != group_name:
                    break
                span.append(j)
                j += 1
            glass_seed = None

        if glass_seed is not None:
            first = glass_seed
            while first > 0 and getattr(surfaces[first - 1], "is_glass", False):
                first -= 1
            last = glass_seed
            while last + 1 < len(surfaces) and getattr(surfaces[last + 1], "is_glass", False):
                last += 1
            if last + 1 < len(surfaces):
                rear = surfaces[last + 1]
                rear_group = groups[last + 1]
                if rear_group and rear_group.get("group_type") == "LENS" and not getattr(rear, "is_glass", False):
                    last += 1
            span = list(range(first, last + 1))

        span = [index for index in span if index not in seen_indices]
        if not span:
            continue
        seen_indices.update(span)
        first_group = groups[span[0]] or grp
        elem_id = first_group.get("element_id", len(elements))
        g_name = first_group.get("group_name", f"Lens {len(elements) + 1}")
        elements.append({
            "element_id": elem_id,
            "group_name": g_name,
            "surface_indices": span,
            "palette": first_group.get("palette"),
            "is_locked": (g_name not in unlocked) and (elem_id not in unlocked),
        })

    # Format human-readable labels
    for elem in elements:
        elem["is_locked"] = all(
            bool(getattr(surfaces[index], "is_locked", True))
            for index in elem["surface_indices"]
        )
        surfs_str = f"S{elem['surface_indices'][0]}" if len(elem['surface_indices']) == 1 else f"S{elem['surface_indices'][0]}-S{elem['surface_indices'][-1]}"
        lock_str = "[LOCKED]" if elem["is_locked"] else "[EDITABLE]"
        elem["label"] = f"{lock_str} {elem['group_name']} ({surfs_str})"

    return elements


def format_surface_material_display(surf: Any, index: int = -1, total_surfaces: int = -1) -> str:
    """
    Format standard Zemax material description for a surface.
    In Zemax OpticStudio, mirrors are designated as 'MIRROR', air spaces as 'AIR',
    and refracting elements by their standard catalog glass name (e.g. 'N-BK7').
    """
    stype = str(getattr(surf, 'surface_type', '')).upper()
    glass = str(getattr(surf, 'glass', '') or "").strip()
    glass_up = glass.upper()

    # Mirror / Reflective surface
    if "MIRROR" in stype or "MIRROR" in glass_up or "MIRR" in glass_up or "REFL" in glass_up or (hasattr(surf, 'is_reflective') and surf.is_reflective()):
        return "MIRROR"

    # Refractive optical glass
    if glass and glass_up not in ("AIR", "NONE", "", "VACUUM"):
        return glass

    return "AIR"


def get_surface_role_description(surf: Any, index: int = -1, total_surfaces: int = -1) -> str:
    """
    Format detailed diagnostic/role description for tooltips and inspect dialogs.
    """
    stype = str(getattr(surf, 'surface_type', '')).upper()
    glass = str(getattr(surf, 'glass', '') or "").strip()
    glass_up = glass.upper()

    # Aperture Stop / Baffle
    if is_stop_plane(surf) or "STOP" in stype:
        return "Diaphragm / Baffle (Aperture Stop)"

    # Detector / Image plane
    if (total_surfaces > 0 and index == total_surfaces - 1) or is_detector_plane(surf, index, total_surfaces) or "DETECTOR" in stype or "IMAGE" in stype:
        return "Sensor Package (Image Plane)"

    # Object / Source plane
    if (index == 0 and is_object_plane(surf, index)) or "OBJECT" in stype:
        return "Stage / Mount (Object Reference)"

    # Diffraction Grating
    if "GRATING" in stype or getattr(surf, 'grating_lines_per_mm', 0) > 0:
        lines = getattr(surf, 'grating_lines_per_mm', 0.0)
        return f"Grating ({lines:g} lp/mm)" if lines > 0 else "Grating Substrate"

    # Mirror
    if "MIRROR" in stype or "MIRROR" in glass_up or "MIRR" in glass_up or "REFL" in glass_up or (hasattr(surf, 'is_reflective') and surf.is_reflective()):
        return "Mirror Substrate (Reflector)"

    # Refractive optical glass
    if glass and glass_up not in ("AIR", "NONE", "", "VACUUM"):
        return f"Refractive Lens Element ({glass})"

    return "Air Gap Spacing"

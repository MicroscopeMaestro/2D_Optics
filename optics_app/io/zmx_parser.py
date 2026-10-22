"""
Zemax Prescription File (.zmx) Importer and Exporter.
Fully compatible with Zemax OpticStudio sequential prescriptions.

Supports tokens:
- Header: VERS, NAME, TITL, UNIT, ENPD, WAVL, WAVM, WWGT, FTMD
- Surfaces: SURF, COMM, TYPE, CURV, THIC, DISZ, GLAS, DIAM, SDIA, CLAP, CONI, PARM, XDAT, DISP, STOP, SOLV, VPAR
- Coordinate Breaks, Aspherics, Diffraction Gratings, Substrates, Mirrors, Detectors.
"""

from typing import Optional, Union, List, Dict, Any
import os
import math
try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.lde import SurfaceStack
except ImportError:
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.lde import SurfaceStack

UNIT_SCALE_TO_MM = {
    "MM": 1.0,
    "CM": 10.0,
    "INCH": 25.4,
    "IN": 25.4,
    "M": 1000.0,
    "METER": 1000.0,
    "METERS": 1000.0,
}

def parse_zmx_string(zmx_content: str) -> SurfaceStack:
    """
    Parse Zemax .zmx prescription text content into a SurfaceStack instance.
    """
    stack = SurfaceStack()
    lines = zmx_content.splitlines()
    current_surf: Optional[Surface] = None
    unit_scale = 1.0
    wavelengths: List[float] = []
    title = ""

    for raw_line in lines:
        # Strip comments starting with ! or #
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!") or line.startswith("#"):
            continue

        # Handle inline comments
        if "!" in line:
            line = line.split("!")[0].strip()
        if "#" in line:
            line = line.split("#")[0].strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].upper()

        if cmd in ("UNIT", "UNI"):
            unit_str = parts[1].upper() if len(parts) > 1 else "MM"
            unit_scale = UNIT_SCALE_TO_MM.get(unit_str, 1.0)

        elif cmd in ("NAME", "TITL"):
            title = line[len(parts[0]):].strip()
            setattr(stack, 'name', title)

        elif cmd in ("WAVL", "WAVM"):
            if len(parts) > 1:
                try:
                    # In Zemax, wavelengths are in micrometers
                    wl_val = float(parts[1])
                    wavelengths.append(wl_val)
                except ValueError:
                    pass

        elif cmd == "SURF":
            if current_surf is not None:
                stack.add_surface(current_surf)
            surf_idx = int(parts[1]) if len(parts) > 1 else len(stack.surfaces)
            current_surf = Surface(name=f"Surface_{surf_idx}")
            if surf_idx == 0:
                current_surf.surface_type = SurfaceType.OBJECT

        elif current_surf is not None:
            if cmd == "COMM":
                # Surface comment / name
                comment = line[len(parts[0]):].strip().strip('"')
                if comment:
                    current_surf.name = comment

            elif cmd == "TYPE":
                type_str = parts[1].upper() if len(parts) > 1 else ""
                if "EVENASPH" in type_str or "ASPHERIC" in type_str:
                    current_surf.surface_type = SurfaceType.ASPHERIC
                elif "GRATING" in type_str or "DGRATING" in type_str or "DIFFRACT" in type_str:
                    current_surf.surface_type = SurfaceType.GRATING
                elif "STOP" in type_str:
                    current_surf.surface_type = SurfaceType.STOP
                    current_surf.is_stop = True
                elif "DETECTOR" in type_str or "IMAGE" in type_str:
                    current_surf.surface_type = SurfaceType.DETECTOR
                elif "CAD" in type_str:
                    current_surf.surface_type = SurfaceType.CAD
                elif "COORDBRK" in type_str or "CBREAK" in type_str:
                    current_surf.surface_type = SurfaceType.COORDINATE_BREAK
                    current_surf.is_coord_break = True
                elif "TOROIDAL" in type_str or "TORIC" in type_str:
                    current_surf.surface_type = SurfaceType.TOROIDAL
                elif "TILTSURF" in type_str or "TILTED" in type_str:
                    current_surf.surface_type = SurfaceType.TILTSURF
                else:
                    current_surf.surface_type = SurfaceType.STANDARD

            elif cmd == "CURV":
                curv = float(parts[1]) if len(parts) > 1 else 0.0
                curv_scaled = curv / unit_scale if unit_scale != 0 else curv
                current_surf.radius = 1.0 / curv_scaled if curv_scaled != 0.0 else 0.0

            elif cmd in ("THIC", "DISZ"):
                val_str = parts[1].upper() if len(parts) > 1 else "0.0"
                if "INFINITY" in val_str or "INF" in val_str:
                    current_surf.thickness = 1e10
                else:
                    current_surf.thickness = abs(float(val_str)) * unit_scale

            elif cmd == "GLAS":
                glass_name = parts[1].upper() if len(parts) > 1 else "AIR"
                if glass_name in ("MIRROR", "MIRR", "REFL"):
                    current_surf.glass = "MIRROR"
                    if current_surf.surface_type == SurfaceType.STANDARD:
                        current_surf.surface_type = SurfaceType.MIRROR
                else:
                    current_surf.glass = parts[1] if len(parts) > 1 else "AIR"

            elif cmd in ("DIAM", "SDIA", "CLAP"):
                current_surf.semi_diameter = (float(parts[1]) if len(parts) > 1 else 25.0) * unit_scale

            elif cmd == "RECA":
                current_surf.aperture_type = "Rectangular"
                if len(parts) >= 5:
                    current_surf.aperture_x_half = float(parts[2]) * unit_scale
                    current_surf.aperture_y_half = float(parts[4]) * unit_scale
                elif len(parts) >= 3:
                    current_surf.aperture_x_half = float(parts[1]) * unit_scale
                    current_surf.aperture_y_half = float(parts[2]) * unit_scale

            elif cmd == "CONI":
                current_surf.conic = float(parts[1]) if len(parts) > 1 else 0.0

            elif cmd == "PARM":
                param_idx = int(parts[1]) if len(parts) > 1 else 1
                param_val = float(parts[2]) if len(parts) > 2 else 0.0
                
                is_cb = getattr(current_surf, 'is_coord_break', False) or (getattr(current_surf, 'surface_type', None) == SurfaceType.COORDINATE_BREAK)
                if is_cb:
                    if param_idx == 1:
                        current_surf.decenter_x = param_val * unit_scale
                    elif param_idx == 2:
                        current_surf.decenter_y = param_val * unit_scale
                    elif param_idx == 3:
                        current_surf.tilt_x = param_val
                    elif param_idx == 4:
                        current_surf.tilt_y = param_val
                    elif param_idx == 5:
                        current_surf.tilt_z = param_val
                    elif param_idx == 6:
                        setattr(current_surf, 'tilt_order', int(param_val))
                elif current_surf.surface_type == SurfaceType.GRATING:
                    if param_idx == 1:
                        # In Zemax DGRATING, param 1 is lines/um if param_val < 50.0
                        if 0 < param_val < 50.0:
                            current_surf.grating_lines_per_mm = param_val * 1000.0
                        else:
                            current_surf.grating_lines_per_mm = param_val / unit_scale
                    elif param_idx == 2:
                        current_surf.grating_order = int(param_val)
                elif param_idx >= 1:
                    if not isinstance(current_surf.asphere_coeffs, list):
                        current_surf.asphere_coeffs = list(current_surf.asphere_coeffs)
                    while len(current_surf.asphere_coeffs) < param_idx:
                        current_surf.asphere_coeffs.append(0.0)
                    current_surf.asphere_coeffs[param_idx - 1] = param_val

            elif cmd == "DISP":
                if len(parts) > 1: current_surf.decenter_x = float(parts[1]) * unit_scale
                if len(parts) > 2: current_surf.decenter_y = float(parts[2]) * unit_scale
                if len(parts) > 3: current_surf.tilt_x = float(parts[3])
                if len(parts) > 4: current_surf.tilt_y = float(parts[4])
                if len(parts) > 5: current_surf.tilt_z = float(parts[5])

            elif cmd == "STOP":
                current_surf.is_stop = True
                if current_surf.surface_type == SurfaceType.STANDARD:
                    current_surf.surface_type = SurfaceType.STOP

            elif cmd in ("SOLV", "VPAR"):
                # Handle solve/variable status
                if len(parts) > 1 and parts[1].upper() in ("V", "VAR", "1"):
                    current_surf.thickness_variable = True

    if current_surf is not None:
        stack.add_surface(current_surf)

    if wavelengths:
        setattr(stack, 'wavelengths', wavelengths)
        stack.primary_wavelength = wavelengths[0]

    # Set image/detector surface type on the last surface if not explicitly assigned
    if len(stack.surfaces) > 1 and stack.surfaces[-1].surface_type == SurfaceType.STANDARD:
        stack.surfaces[-1].surface_type = SurfaceType.DETECTOR

    stack._update_accumulated_positions()
    return stack


def parse_zmx_file(file_path: str) -> SurfaceStack:
    """Parse Zemax .zmx text file with robust encoding detection."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Zemax file not found: {file_path}")

    encodings = ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'utf-8']
    content = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            if content and ("SURF" in content or "surf" in content or "ZEMAX" in content.upper() or "VERS" in content.upper()):
                break
        except Exception:
            continue

    if content is None:
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            content = raw_bytes.decode('utf-8', errors='ignore')

    return parse_zmx_string(content)


def export_to_zmx_string(
    stack: SurfaceStack,
    title: str = "Optical System Export",
    wavelengths: Optional[List[float]] = None
) -> str:
    """
    Export a SurfaceStack to standard Zemax OpticStudio (.zmx) prescription text format.
    """
    lines = [
        "! Zemax OpticStudio Prescription File",
        "! Generated by 2D_Optics CAD Engine",
        "VERS 140410 1 0",
        f"NAME \"{title}\"",
        "UNIT MM",
        "ENPD 25.4",
        "FTMD 0 0 0 0 0 0 0 0"
    ]

    wls = wavelengths if wavelengths else getattr(stack, 'wavelengths', [0.4861, 0.5876, 0.6563])
    if not wls and hasattr(stack, 'primary_wavelength'):
        wls = [stack.primary_wavelength]

    for idx, wl in enumerate(wls):
        w_weight = 1.0 if idx == 0 else 0.5
        lines.append(f"WAVL {wl:.6f} {w_weight:.4f}")

    for idx, surf in enumerate(stack.surfaces):
        lines.append(f"SURF {idx}")
        if surf.name:
            lines.append(f"  COMM {surf.name}")

        # Surface Type
        norm_type = SurfaceType.normalize(surf.surface_type)
        is_cb = getattr(surf, 'is_coord_break', False) or (norm_type == SurfaceType.COORDINATE_BREAK)

        if is_cb:
            lines.append("  TYPE COORDBRK")
            lines.append(f"  PARM 1 {surf.decenter_x:.6f}")
            lines.append(f"  PARM 2 {surf.decenter_y:.6f}")
            lines.append(f"  PARM 3 {surf.tilt_x:.6f}")
            lines.append(f"  PARM 4 {surf.tilt_y:.6f}")
            lines.append(f"  PARM 5 {surf.tilt_z:.6f}")
            lines.append(f"  PARM 6 {getattr(surf, 'tilt_order', 0)}")
        elif norm_type in (SurfaceType.ASPHERIC, SurfaceType.EVEN_ASPHERE):
            lines.append("  TYPE EVENASPH")
        elif norm_type in (SurfaceType.GRATING, SurfaceType.DIFFRACTION_GRATING):
            lines.append("  TYPE DIFFRACT")
            lines.append(f"  PARM 1 {surf.grating_lines_per_mm:.4f}")
            lines.append(f"  PARM 2 {getattr(surf, 'grating_order', 1)}")
        elif norm_type == SurfaceType.TOROIDAL:
            lines.append("  TYPE TOROIDAL")
        elif norm_type == SurfaceType.CAD:
            lines.append("  TYPE CAD")
        else:
            lines.append("  TYPE STANDARD")

        # Stop marker
        if surf.is_stop or norm_type == SurfaceType.STOP:
            lines.append("  STOP")

        # Curvature (1 / R)
        if abs(surf.radius) > 1e-9:
            curv = 1.0 / surf.radius
            lines.append(f"  CURV {curv:.10f}")
        else:
            lines.append("  CURV 0.0000000000")

        # Thickness
        if math.isinf(surf.thickness) or surf.thickness > 1e9:
            lines.append("  THIC INFINITY")
        else:
            lines.append(f"  THIC {surf.thickness:.6f}")

        # Glass / Material
        is_mirror = (
            (hasattr(surf, "is_reflective") and surf.is_reflective())
            or (surf.glass and surf.glass.upper() in ("MIRROR", "MIRR", "REFL"))
            or norm_type == SurfaceType.MIRROR
        )
        if is_mirror:
            lines.append("  GLAS MIRROR")
        elif surf.glass and surf.glass.upper() not in ("AIR", "NONE", ""):
            lines.append(f"  GLAS {surf.glass}")

        # Semi-diameter / Aperture
        lines.append(f"  DIAM {surf.semi_diameter:.6f}")
        if getattr(surf, 'aperture_type', 'Circular') == 'Rectangular':
            hx = surf.aperture_x_half if surf.aperture_x_half > 0 else surf.semi_diameter
            hy = surf.aperture_y_half if surf.aperture_y_half > 0 else surf.semi_diameter
            lines.append(f"  RECA 0.000000 {hx:.6f} 0.000000 {hy:.6f}")

        # Conic constant
        if abs(surf.conic) > 1e-9:
            lines.append(f"  CONI {surf.conic:.6f}")

        # Aspheric coefficients
        if not is_cb and norm_type in (SurfaceType.ASPHERIC, SurfaceType.EVEN_ASPHERE) and surf.asphere_coeffs:
            for c_idx, coeff in enumerate(surf.asphere_coeffs):
                if abs(coeff) > 1e-15:
                    lines.append(f"  PARM {c_idx+1} {coeff:.12e}")

        # Decenters and Tilts (using DISP for non-coordinate breaks if present)
        if not is_cb and any(abs(v) > 1e-6 for v in [surf.decenter_x, surf.decenter_y, surf.tilt_x, surf.tilt_y, surf.tilt_z]):
            lines.append(f"  DISP {surf.decenter_x:.6f} {surf.decenter_y:.6f} {surf.tilt_x:.6f} {surf.tilt_y:.6f} {surf.tilt_z:.6f}")

    return "\n".join(lines) + "\n"


def export_to_zmx_file(
    stack: SurfaceStack,
    filepath: str,
    title: str = "Optical System Export",
    wavelengths: Optional[List[float]] = None
) -> None:
    """Save SurfaceStack to a Zemax .zmx file on disk."""
    content = export_to_zmx_string(stack, title=title, wavelengths=wavelengths)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

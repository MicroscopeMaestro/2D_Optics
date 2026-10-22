"""
2D CAD Technical Drawing SVG Exporter for 2D_Optics Simulation & Design Studio.
Exports optical systems as fully dimensioned, publication-quality 2D vector CAD drawings
readable in FreeCAD (TechDraw / Draft), Inkscape, and web browsers.
"""

import os
import math
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


class SVG2DExporter:
    """
    Exports 2D cross-sectional optical drawings as SVG with ISO/ANSI dimensions,
    centerlines, lens hatching, ray paths, and optical prescription tables.
    """

    def __init__(self, system_or_stack: Any, folded: bool = True):
        if hasattr(system_or_stack, 'get_surface_stack'):
            self.stack: SurfaceStack = system_or_stack.get_surface_stack()
        elif hasattr(system_or_stack, 'surfaces'):
            self.stack: SurfaceStack = system_or_stack
        else:
            raise ValueError("Invalid optical system or surface stack provided.")

        self.stack._update_accumulated_positions()
        self.folded: bool = folded

    def export(
        self,
        file_path: str,
        include_dimensions: bool = True,
        include_rays: bool = True,
        include_hatches: bool = True,
        include_table: bool = True,
        wavelengths: Optional[List[float]] = None,
        scale: float = 3.5  # pixels per mm
    ) -> str:
        """Export 2D CAD SVG vector drawing with dimensions."""
        wls = wavelengths or [0.785, 0.850, 0.900]

        # Calculate bounding box in mm
        raw_pts = []
        for s in self.stack.surfaces:
            sd = s.semi_diameter
            p1 = self._transform_point(s, float(sd), 0.0)
            p2 = self._transform_point(s, float(-sd), 0.0)
            raw_pts.extend([p1, p2])

        min_x = min(p[0] for p in raw_pts) - 30.0
        max_x = max(p[0] for p in raw_pts) + 30.0
        min_y = min(p[1] for p in raw_pts) - 50.0
        max_y = max(p[1] for p in raw_pts) + 30.0

        if include_table:
            min_y -= 80.0  # Extra space for prescription table

        width_mm = max_x - min_x
        height_mm = max_y - min_y

        viewbox_w = width_mm * scale
        viewbox_h = height_mm * scale

        def to_svg_x(mm_x: float) -> float:
            return (mm_x - min_x) * scale

        def to_svg_y(mm_y: float) -> float:
            # SVG y-axis is inverted (0 at top)
            return (max_y - mm_y) * scale

        svg_lines = [
            f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{viewbox_w:.1f}px" height="{viewbox_h:.1f}px" viewBox="0 0 {viewbox_w:.1f} {viewbox_h:.1f}">',
            f'<style>',
            f'  .lens-outline {{ fill: #e8f4fc; fill-opacity: 0.7; stroke: #1e3d59; stroke-width: 1.5px; }}',
            f'  .component-outline {{ fill: #fff3cd; stroke: #856404; stroke-width: 1.5px; }}',
            f'  .optical-axis {{ stroke: #d9534f; stroke-width: 1.0px; stroke-dasharray: 6,3,1,3; }}',
            f'  .dim-line {{ stroke: #0275d8; stroke-width: 1.0px; }}',
            f'  .dim-text {{ fill: #0275d8; font-family: sans-serif; font-size: 10px; font-weight: bold; text-anchor: middle; }}',
            f'  .callout-text {{ fill: #6f42c1; font-family: sans-serif; font-size: 9px; font-weight: bold; }}',
            f'  .table-header {{ fill: #343a40; font-family: sans-serif; font-size: 10px; font-weight: bold; }}',
            f'  .table-cell {{ fill: #495057; font-family: sans-serif; font-size: 9px; }}',
            f'  .ray-785 {{ stroke: #dc3545; stroke-width: 0.8px; stroke-opacity: 0.8; fill: none; }}',
            f'  .ray-850 {{ stroke: #28a745; stroke-width: 0.8px; stroke-opacity: 0.8; fill: none; }}',
            f'  .ray-900 {{ stroke: #007bff; stroke-width: 0.8px; stroke-opacity: 0.8; fill: none; }}',
            f'</style>',
            f'<defs>',
            f'  <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            f'    <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#0275d8" />',
            f'  </marker>',
            f'</defs>',
            f'<rect width="100%" height="100%" fill="#ffffff" />',
        ]

        # 1. Draw Optical Axis
        for i in range(len(self.stack.surfaces) - 1):
            s1 = self.stack.surfaces[i]
            s2 = self.stack.surfaces[i + 1]
            p1 = self._transform_point(s1, 0.0, 0.0)
            p2 = self._transform_point(s2, 0.0, 0.0)
            x1, y1 = to_svg_x(p1[0]), to_svg_y(p1[1])
            x2, y2 = to_svg_x(p2[0]), to_svg_y(p2[1])
            svg_lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="optical-axis" />')

        # 2. Draw Rays
        if include_rays:
            from optics_app.engine.ray import Ray
            opt_sys = OpticalSystem(self.stack)
            ray_classes = {0: "ray-785", 1: "ray-850", 2: "ray-900"}
            d0 = self.stack.surfaces[0].thickness
            r1 = self.stack.surfaces[1].semi_diameter if len(self.stack.surfaces) > 1 else 10.0

            for wl_idx, wl in enumerate(wls):
                r_cls = ray_classes.get(wl_idx % 3, "ray-850")
                for py in np.linspace(-0.8, 0.8, 5):
                    if d0 > 1e4 or d0 <= 0:
                        start_pos = np.array([0.0, py * r1, 0.0])
                        start_dir = np.array([0.0, 0.0, 1.0])
                    else:
                        start_pos = np.array([0.0, 0.0, 0.0])
                        target_y = py * r1
                        target_z = d0
                        length = np.sqrt(target_y**2 + target_z**2)
                        start_dir = np.array([0.0, target_y / length, target_z / length])
                    
                    r = Ray(pos=start_pos, dir=start_dir, wavelength=wl)
                    res = opt_sys.trace_ray(r)
                    if res and len(res.history) > 1:
                        poly_pts = []
                        for h in res.history:
                            x = h.get("x_global", h.get("x", 0.0))
                            y = h.get("y_global", h.get("y", 0.0))
                            z = h.get("z_global", h.get("z", 0.0))
                            # Projected 2D point (Z, X)
                            mm_pt = (float(z), float(x if abs(x) > abs(y) else y))
                            poly_pts.append(f"{to_svg_x(mm_pt[0]):.1f},{to_svg_y(mm_pt[1]):.1f}")
                        svg_lines.append(f'<polyline points="{" ".join(poly_pts)}" class="{r_cls}" />')

        # 3. Draw Solid Lenses & Planar Components
        i = 0
        while i < len(self.stack.surfaces):
            surf = self.stack.surfaces[i]

            if surf.is_glass and i < len(self.stack.surfaces) - 1:
                next_surf = self.stack.surfaces[i + 1]
                pts_2d = self._generate_lens_contour(surf, next_surf)
                svg_poly = " ".join([f"{to_svg_x(px):.1f},{to_svg_y(py):.1f}" for px, py in pts_2d])
                svg_lines.append(f'<polygon points="{svg_poly}" class="lens-outline" />')

            elif surf.is_reflective() or surf.surface_type in [SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.OBJECT] or "DMD" in (surf.name or "").upper():
                comp_pts = self._generate_planar_component_contour(surf)
                svg_poly = " ".join([f"{to_svg_x(px):.1f},{to_svg_y(py):.1f}" for px, py in comp_pts])
                svg_lines.append(f'<polygon points="{svg_poly}" class="component-outline" />')

            i += 1

        # 4. Draw Dimensions
        if include_dimensions:
            for i, surf in enumerate(self.stack.surfaces):
                is_air = surf.is_air
                ct = surf.thickness
                sd = surf.semi_diameter
                rad = surf.radius

                # Thickness / Distance Dimension
                if ct > 0.1 and i < len(self.stack.surfaces) - 1:
                    next_surf = self.stack.surfaces[i + 1]
                    p_v1 = self._transform_point(surf, 0.0, 0.0)
                    p_v2 = self._transform_point(next_surf, 0.0, 0.0)

                    offset_mm = 14.0 if is_air else 9.0
                    tag = "Dist" if is_air else "Tc"
                    txt = f"{tag}={ct:.1f}"

                    x1, y1 = to_svg_x(p_v1[0]), to_svg_y(p_v1[1] + offset_mm)
                    x2, y2 = to_svg_x(p_v2[0]), to_svg_y(p_v2[1] + offset_mm)
                    tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - 4.0

                    svg_lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="dim-line" marker-start="url(#arrow)" marker-end="url(#arrow)" />')
                    svg_lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="dim-text">{txt}</text>')

                # Diameter Dimension
                if sd > 0.1:
                    p_top = self._transform_point(surf, float(sd), 0.0)
                    p_bot = self._transform_point(surf, float(-sd), 0.0)
                    x_t, y_t = to_svg_x(p_top[0]), to_svg_y(p_top[1])
                    x_b, y_b = to_svg_x(p_bot[0]), to_svg_y(p_bot[1])
                    svg_lines.append(f'<line x1="{x_t:.1f}" y1="{y_t:.1f}" x2="{x_b:.1f}" y2="{y_b:.1f}" stroke="#6c757d" stroke-width="0.8" stroke-dasharray="2,2" />')
                    svg_lines.append(f'<text x="{x_t - 6:.1f}" y="{(y_t + y_b)/2:.1f}" class="callout-text">Ø{2*sd:.1f}</text>')

        # 5. Draw Prescription Table
        if include_table:
            tbl_x_mm = min_x + 15.0
            tbl_y_mm = min_y + 35.0
            x_start = to_svg_x(tbl_x_mm)
            y_start = to_svg_y(tbl_y_mm)

            headers = ["Surf", "Name", "Type", "Radius", "Thickness", "Glass", "Semi-Diam"]
            col_w = [40, 110, 75, 75, 75, 75, 75]

            # Header row
            svg_lines.append(f'<text x="{x_start}" y="{y_start - 12}" class="table-header" style="font-size:12px;">OPTICAL PRESCRIPTION DATA TABLE (2D CAD)</text>')
            cx = x_start
            for c_idx, h in enumerate(headers):
                svg_lines.append(f'<text x="{cx + 4}" y="{y_start + 12}" class="table-header">{h}</text>')
                cx += col_w[c_idx]

            cy = y_start + 18
            for i, s in enumerate(self.stack.surfaces):
                cx = x_start
                stype = str(s.surface_type) if hasattr(s.surface_type, 'value') else s.surface_type
                vals = [
                    f"S{i}",
                    str(s.name)[:14],
                    stype[:10],
                    f"{s.radius:.2f}" if abs(s.radius) > 1e-4 else "Infinity",
                    f"{s.thickness:.2f}",
                    str(s.glass),
                    f"{s.semi_diameter:.2f}"
                ]
                for c_idx, v in enumerate(vals):
                    svg_lines.append(f'<text x="{cx + 4}" y="{cy + 12}" class="table-cell">{v}</text>')
                    cx += col_w[c_idx]
                cy += 16

        svg_lines.append('</svg>')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        return file_path

    def _transform_point(self, surf: Surface, local_x: float, local_z: float) -> Tuple[float, float]:
        if not self.folded:
            z_unfolded = surf.global_z + local_z
            return (float(z_unfolded), float(local_x))

        local_3d = np.array([local_x, 0.0, local_z], dtype=float)
        global_3d = surf.global_origin + surf.global_R @ local_3d
        return (float(global_3d[2]), float(global_3d[0]))

    def _generate_lens_contour(self, surf1: Surface, surf2: Surface, num_pts: int = 35) -> List[Tuple[float, float]]:
        sd1 = surf1.semi_diameter
        sd2 = surf2.semi_diameter
        sd = max(sd1, sd2)
        ct = surf1.thickness
        pts = []

        # Front
        for r in np.linspace(sd, -sd, num_pts):
            r_c = np.clip(abs(r), 0, sd1)
            pts.append(self._transform_point(surf1, float(r), float(surf1.sag(r_c))))

        # Bottom
        pts.append(self._transform_point(surf1, float(-sd), float(ct + surf2.sag(min(sd, sd2)))))

        # Rear
        for r in np.linspace(-sd, sd, num_pts):
            r_c = np.clip(abs(r), 0, sd2)
            pts.append(self._transform_point(surf1, float(r), float(ct + surf2.sag(r_c))))

        # Top
        pts.append(self._transform_point(surf1, float(sd), float(surf1.sag(min(sd, sd1)))))

        return pts

    def _generate_planar_component_contour(self, surf: Surface) -> List[Tuple[float, float]]:
        if getattr(surf, 'aperture_type', '') == "Rectangular":
            sd = surf.aperture_x_half if getattr(surf, 'aperture_x_half', 0) > 0 else surf.semi_diameter
        else:
            sd = surf.semi_diameter

        sub_thick = float(getattr(surf, 'substrate_thickness', 6.0) if getattr(surf, 'substrate_thickness', 0) > 0 else 6.0)
        if surf.surface_type == SurfaceType.OBJECT:
            sd = max(sd, 1.0)
            sub_thick = 0.5
        elif surf.surface_type == SurfaceType.DETECTOR:
            sub_thick = float(getattr(surf, 'substrate_thickness', 1.5) if getattr(surf, 'substrate_thickness', 0) > 0 else 1.5)

        return [
            self._transform_point(surf, float(sd), 0.0),
            self._transform_point(surf, float(-sd), 0.0),
            self._transform_point(surf, float(-sd), float(-sub_thick)),
            self._transform_point(surf, float(sd), float(-sub_thick)),
        ]

"""
2D CAD DXF Exporter for 2D_Optics Simulation & Design Studio.
Exports optical systems as fully dimensioned 2D CAD models (DXF R2010 / R2000)
with multi-plane support: XZ (Dispersion/Horizontal), YZ/YX (Elevation/Vertical),
XY (En-Face Apertures), and DUAL multi-view projection for FreeCAD, AutoCAD, and SolidWorks.
"""

import os
import math
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

try:
    import ezdxf
    from ezdxf import colors
    from ezdxf.enums import TextEntityAlignment
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


class DXF2DExporter:
    """
    Exports 2D cross-sectional optical CAD drawings with ISO/ANSI dimensions,
    centerlines, lens cross-hatching, multi-plane (XZ, YZ/YX, XY, DUAL) views,
    ray trajectories, and prescription tables.
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
        plane: str = "DUAL"  # "XZ", "YZ", "XY", "YX", or "DUAL"
    ) -> str:
        """
        Export 2D CAD DXF model with dimensions across selected projection plane(s).
        """
        if not HAS_EZDXF:
            return self._export_pure_python_dxf(file_path, include_dimensions, include_rays)

        doc = ezdxf.new('R2010', setup=True)
        msp = doc.modelspace()

        # Setup standard CAD Layers
        self._setup_layers(doc)

        plane_mode = plane.upper()

        if plane_mode in ["DUAL", "BOTH", "ALL"]:
            # Dual view: XZ plane on top, YZ/YX plane below
            # 1. Draw XZ Plane (Dispersion / Horizontal)
            bounds_xz = self._draw_plane_view(
                doc, msp,
                plane="XZ",
                offset_y=60.0,
                view_title="SECTION A-A: XZ PLANE (DISPERSION / HORIZONTAL VIEW) [MM]",
                include_dimensions=include_dimensions,
                include_rays=include_rays,
                include_hatches=include_hatches,
                wavelengths=wavelengths
            )

            # 2. Draw YZ / YX Plane (Elevation / Vertical)
            bounds_yz = self._draw_plane_view(
                doc, msp,
                plane="YZ",
                offset_y=-60.0,
                view_title="SECTION B-B: YZ / YX PLANE (ELEVATION / SLIT VIEW) [MM]",
                include_dimensions=include_dimensions,
                include_rays=include_rays,
                include_hatches=include_hatches,
                wavelengths=wavelengths
            )

            combined_bounds = {
                "min_x": min(bounds_xz["min_x"], bounds_yz["min_x"]),
                "max_x": max(bounds_xz["max_x"], bounds_yz["max_x"]),
                "min_y": min(bounds_xz["min_y"], bounds_yz["min_y"]),
                "max_y": max(bounds_xz["max_y"], bounds_yz["max_y"]),
            }

            if include_table:
                self._draw_prescription_table(msp, combined_bounds)

        else:
            # Single plane view (XZ, YZ, or XY)
            p_name = "XZ PLANE (DISPERSION / HORIZONTAL)" if "X" in plane_mode and "Z" in plane_mode else ("YZ / YX PLANE (ELEVATION / VERTICAL)" if "Y" in plane_mode else "XY PLANE")
            bounds = self._draw_plane_view(
                doc, msp,
                plane=plane_mode,
                offset_y=0.0,
                view_title=f"2D CROSS-SECTION: {p_name} [MM]",
                include_dimensions=include_dimensions,
                include_rays=include_rays,
                include_hatches=include_hatches,
                wavelengths=wavelengths
            )
            if include_table:
                self._draw_prescription_table(msp, bounds)

        doc.saveas(file_path)
        return file_path

    def _setup_layers(self, doc):
        """Define standard CAD layers with colors and linetypes."""
        layers = [
            ("OPTICS_LENSES", colors.WHITE, "CONTINUOUS", 0.50),
            ("OPTICS_COMPONENTS", colors.YELLOW, "CONTINUOUS", 0.35),
            ("OPTICS_HATCH", colors.GRAY, "CONTINUOUS", 0.13),
            ("OPTICAL_AXIS", colors.RED, "CENTER", 0.25),
            ("DIMENSIONS_LINEAR", colors.CYAN, "CONTINUOUS", 0.25),
            ("DIMENSIONS_RADIAL", colors.MAGENTA, "CONTINUOUS", 0.25),
            ("ANNOTATIONS", colors.GREEN, "CONTINUOUS", 0.25),
            ("RAYS_785NM", colors.RED, "CONTINUOUS", 0.18),
            ("RAYS_850NM", colors.GREEN, "CONTINUOUS", 0.18),
            ("RAYS_900NM", colors.BLUE, "CONTINUOUS", 0.18),
            ("TITLE_BLOCK", colors.WHITE, "CONTINUOUS", 0.35),
        ]
        for name, color, linetype, lineweight in layers:
            if name not in doc.layers:
                doc.layers.add(
                    name=name,
                    color=color,
                    linetype=linetype,
                    lineweight=int(lineweight * 100)
                )

    def _transform_point(
        self,
        surf: Surface,
        local_h: float,
        local_z: float,
        plane: str = "XZ",
        offset: Tuple[float, float] = (0.0, 0.0)
    ) -> Tuple[float, float]:
        """
        Transforms local 2D lens cross-section point (h, z) to 2D CAD drawing coordinates.
        Supports XZ (Z vs X), YZ (Z vs Y), and XY (X vs Y).
        """
        is_yz = "Y" in plane.upper() and "Z" in plane.upper()
        is_xy = "X" in plane.upper() and "Y" in plane.upper()

        if not self.folded:
            # Unfolded mode: straight optical axis along drawing X-axis, semi-diameter along drawing Y-axis
            z_unfolded = surf.global_z + local_z
            return (float(z_unfolded + offset[0]), float(local_h + offset[1]))

        # Folded mode: 3D global position projected to selected 2D CAD plane
        if is_yz:
            local_3d = np.array([0.0, local_h, local_z], dtype=float)
            global_3d = surf.global_origin + surf.global_R @ local_3d
            return (float(global_3d[2] + offset[0]), float(global_3d[1] + offset[1]))
        elif is_xy:
            local_3d = np.array([local_h, local_z, 0.0], dtype=float)
            global_3d = surf.global_origin + surf.global_R @ local_3d
            return (float(global_3d[0] + offset[0]), float(global_3d[1] + offset[1]))
        else:
            # Default XZ Plane (Z_CAD, X_CAD)
            local_3d = np.array([local_h, 0.0, local_z], dtype=float)
            global_3d = surf.global_origin + surf.global_R @ local_3d
            return (float(global_3d[2] + offset[0]), float(global_3d[0] + offset[1]))

    def _draw_plane_view(
        self,
        doc,
        msp,
        plane: str,
        offset_y: float,
        view_title: str,
        include_dimensions: bool,
        include_rays: bool,
        include_hatches: bool,
        wavelengths: Optional[List[float]]
    ) -> Dict[str, float]:
        """Draw complete projected view for a specific plane (XZ or YZ)."""
        offset = (0.0, offset_y)
        min_x, max_x = 1e9, -1e9
        min_y, max_y = 1e9, -1e9

        # 1. Draw Optical Elements
        i = 0
        while i < len(self.stack.surfaces):
            surf = self.stack.surfaces[i]

            # Lens Solid Element
            if surf.is_glass and i < len(self.stack.surfaces) - 1:
                next_surf = self.stack.surfaces[i + 1]
                pts_2d = self._generate_lens_contour(surf, next_surf, plane=plane, offset=offset)
                
                msp.add_lwpolyline(pts_2d, close=True, dxfattribs={"layer": "OPTICS_LENSES"})
                
                if include_hatches and len(pts_2d) > 3:
                    try:
                        hatch = msp.add_hatch(color=8, dxfattribs={"layer": "OPTICS_HATCH"})
                        hatch.set_pattern_fill("ANSI31", scale=2.5, angle=45)
                        hatch.paths.add_polyline_path(pts_2d, is_closed=True)
                    except Exception:
                        pass

                for px, py in pts_2d:
                    min_x, max_x = min(min_x, px), max(max_x, px)
                    min_y, max_y = min(min_y, py), max(max_y, py)

            # Planar Component (Grating, Mirror, DMD, Detector)
            elif surf.is_reflective() or surf.surface_type in [SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.OBJECT] or "DMD" in (surf.name or "").upper():
                comp_pts = self._generate_planar_component_contour(surf, plane=plane, offset=offset)
                layer = "OPTICS_COMPONENTS" if surf.surface_type != SurfaceType.OBJECT else "ANNOTATIONS"
                msp.add_lwpolyline(comp_pts, close=True, dxfattribs={"layer": layer})
                
                # Add substrate cross-hatch pattern for mirrors, gratings, and DMDs
                if include_hatches and (surf.is_reflective() or surf.surface_type == SurfaceType.GRATING or "DMD" in (surf.name or "").upper()) and len(comp_pts) > 3:
                    try:
                        hatch = msp.add_hatch(color=9, dxfattribs={"layer": "OPTICS_HATCH"})
                        hatch.set_pattern_fill("ANSI37", scale=3.0, angle=45)
                        hatch.paths.add_polyline_path(comp_pts, is_closed=True)
                    except Exception:
                        pass

                for px, py in comp_pts:
                    min_x, max_x = min(min_x, px), max(max_x, px)
                    min_y, max_y = min(min_y, py), max(max_y, py)

            i += 1

        # 2. Draw Optical Axis
        self._draw_optical_axis(msp, plane=plane, offset=offset)

        # 3. Draw Rays
        if include_rays:
            self._draw_ray_paths(msp, plane=plane, offset=offset, wavelengths=wavelengths)

        # 4. Draw Dimensions
        if include_dimensions:
            self._draw_dimensions(doc, msp, plane=plane, offset=offset)

        # 5. Add View Title Label & Scale Reference
        title_x = min_x if min_x != 1e9 else 0.0
        title_y = (max_y + 12.0) if max_y != -1e9 else (offset_y + 25.0)

        msp.add_text(
            view_title,
            height=3.0,
            dxfattribs={"layer": "ANNOTATIONS"}
        ).set_placement((title_x, title_y), align=TextEntityAlignment.BOTTOM_LEFT)

        # Scale bar (50 mm reference bar)
        sb_start = (title_x, title_y - 4.0)
        sb_end = (title_x + 50.0, title_y - 4.0)
        msp.add_line(sb_start, sb_end, dxfattribs={"layer": "ANNOTATIONS"})
        msp.add_line((sb_start[0], sb_start[1] - 1.5), (sb_start[0], sb_start[1] + 1.5), dxfattribs={"layer": "ANNOTATIONS"})
        msp.add_line((sb_end[0], sb_end[1] - 1.5), (sb_end[0], sb_end[1] + 1.5), dxfattribs={"layer": "ANNOTATIONS"})
        msp.add_text(
            "SCALE: 50.0 mm",
            height=1.8,
            dxfattribs={"layer": "ANNOTATIONS"}
        ).set_placement(((sb_start[0] + sb_end[0]) / 2.0, sb_start[1] + 1.0), align=TextEntityAlignment.BOTTOM_CENTER)

        return {
            "min_x": min_x if min_x != 1e9 else 0.0,
            "max_x": max_x if max_x != -1e9 else 100.0,
            "min_y": min_y if min_y != 1e9 else (offset_y - 20.0),
            "max_y": max_y if max_y != -1e9 else (offset_y + 20.0),
        }

    def _generate_lens_contour(
        self,
        surf1: Surface,
        surf2: Surface,
        plane: str = "XZ",
        offset: Tuple[float, float] = (0.0, 0.0),
        num_pts: int = 40
    ) -> List[Tuple[float, float]]:
        """Compute exact 2D closed polygon coordinates for a solid lens element."""
        sd1 = surf1.semi_diameter
        sd2 = surf2.semi_diameter
        sd = max(sd1, sd2)
        ct = surf1.thickness

        pts = []

        # Front surface profile: from +sd to -sd
        r_front = np.linspace(sd, -sd, num_pts)
        for r in r_front:
            r_clamped = np.clip(abs(r), 0, sd1)
            z_local = surf1.sag(r_clamped)
            pts.append(self._transform_point(surf1, float(r), float(z_local), plane=plane, offset=offset))

        # Bottom edge flat: connect front to rear
        z_rear_bot = ct + surf2.sag(min(sd, sd2))
        pts.append(self._transform_point(surf1, float(-sd), float(z_rear_bot), plane=plane, offset=offset))

        # Rear surface profile: from -sd back to +sd
        r_rear = np.linspace(-sd, sd, num_pts)
        for r in r_rear:
            r_clamped = np.clip(abs(r), 0, sd2)
            z_local = ct + surf2.sag(r_clamped)
            pts.append(self._transform_point(surf1, float(r), float(z_local), plane=plane, offset=offset))

        # Top edge flat: connect rear back to front
        z_front_top = surf1.sag(min(sd, sd1))
        pts.append(self._transform_point(surf1, float(sd), float(z_front_top), plane=plane, offset=offset))

        return pts

    def _generate_planar_component_contour(
        self,
        surf: Surface,
        plane: str = "XZ",
        offset: Tuple[float, float] = (0.0, 0.0)
    ) -> List[Tuple[float, float]]:
        """Compute 2D cross-section rectangle for planar optical components with substrate thickness."""
        is_yz = "Y" in plane.upper() and "Z" in plane.upper()
        if getattr(surf, 'aperture_type', '') == "Rectangular":
            sd = surf.aperture_y_half if is_yz else surf.aperture_x_half
        else:
            sd = surf.semi_diameter

        sub_thick = float(getattr(surf, 'substrate_thickness', 6.0) if getattr(surf, 'substrate_thickness', 0) > 0 else 6.0)

        if surf.surface_type == SurfaceType.OBJECT:
            sd = max(sd, 1.0)
            sub_thick = 0.5
        elif surf.surface_type == SurfaceType.DETECTOR:
            sub_thick = float(getattr(surf, 'substrate_thickness', 1.5) if getattr(surf, 'substrate_thickness', 0) > 0 else 1.5)

        p1 = self._transform_point(surf, float(sd), 0.0, plane=plane, offset=offset)
        p2 = self._transform_point(surf, float(-sd), 0.0, plane=plane, offset=offset)
        p3 = self._transform_point(surf, float(-sd), float(-sub_thick), plane=plane, offset=offset)
        p4 = self._transform_point(surf, float(sd), float(-sub_thick), plane=plane, offset=offset)

        return [p1, p2, p3, p4]

    def _draw_optical_axis(self, msp, plane: str = "XZ", offset: Tuple[float, float] = (0.0, 0.0)):
        """Draw dashed-dot centerline along optical axis between surfaces."""
        pts = []
        for s in self.stack.surfaces:
            pts.append(self._transform_point(s, 0.0, 0.0, plane=plane, offset=offset))

        for k in range(len(pts) - 1):
            msp.add_line(pts[k], pts[k+1], dxfattribs={"layer": "OPTICAL_AXIS"})

    def _draw_ray_paths(
        self,
        msp,
        plane: str = "XZ",
        offset: Tuple[float, float] = (0.0, 0.0),
        wavelengths: Optional[List[float]] = None
    ):
        """Traces and draws multi-spectral ray trajectories projected onto plane."""
        from optics_app.engine.ray import Ray

        wls = wavelengths or [0.785, 0.850, 0.900]
        layer_map = {
            0: "RAYS_785NM",
            1: "RAYS_850NM",
            2: "RAYS_900NM",
        }

        opt_sys = OpticalSystem(self.stack)
        is_yz = "Y" in plane.upper() and "Z" in plane.upper()

        for w_idx, wl in enumerate(wls):
            layer_name = layer_map.get(w_idx % 3, "RAYS_785NM")
            
            # Trace marginal, pupil, and chief rays
            pupil_positions = [-1.0, -0.6, 0.0, 0.6, 1.0]
            r1 = self.stack.surfaces[1].semi_diameter if len(self.stack.surfaces) > 1 else 6.35
            d0 = self.stack.surfaces[0].thickness

            for p_coord in pupil_positions:
                if d0 <= 1e-4:
                    if is_yz:
                        start_pos = np.array([0.0, p_coord * r1 * 0.85, 0.0])
                    else:
                        start_pos = np.array([p_coord * r1 * 0.85, 0.0, 0.0])
                    start_dir = np.array([0.0, 0.0, 1.0])
                else:
                    start_pos = np.array([0.0, 0.0, 0.0])
                    target_h = p_coord * r1 * 0.85
                    target_z = d0
                    length = np.sqrt(target_h**2 + target_z**2)
                    if is_yz:
                        start_dir = np.array([0.0, target_h / length, target_z / length])
                    else:
                        start_dir = np.array([target_h / length, 0.0, target_z / length])
                
                r = Ray(pos=start_pos, dir=start_dir, wavelength=wl)
                ray_res = opt_sys.trace_ray(r)
                
                if ray_res and len(ray_res.history) > 1:
                    ray_pts_2d = []
                    for pt in ray_res.history:
                        xg = pt.get("x_global", pt.get("x", 0.0))
                        yg = pt.get("y_global", pt.get("y", 0.0))
                        zg = pt.get("z_global", pt.get("z", 0.0))
                        
                        if is_yz:
                            ray_pts_2d.append((float(zg + offset[0]), float(yg + offset[1])))
                        else:
                            ray_pts_2d.append((float(zg + offset[0]), float(xg + offset[1])))
                    
                    for k in range(len(ray_pts_2d) - 1):
                        msp.add_line(ray_pts_2d[k], ray_pts_2d[k+1], dxfattribs={"layer": layer_name})

    def _draw_dimensions(
        self,
        doc,
        msp,
        plane: str = "XZ",
        offset: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Generates standard ISO/ANSI engineering dimensions with witness lines,
        arrowheads, and callout text for center thicknesses, air gaps, diameters, and radii.
        """
        dim_style = "EZ_OPTICS_DIM"
        if dim_style not in doc.dimstyles:
            ds = doc.dimstyles.new(dim_style)
            ds.dxf.dimtxt = 2.0     # Text height 2mm
            ds.dxf.dimasz = 2.0     # Arrow size 2mm
            ds.dxf.dimexe = 1.0     # Extension line extension
            ds.dxf.dimexo = 1.0     # Extension line offset
            ds.dxf.dimdec = 2       # 2 decimal places

        for i, surf in enumerate(self.stack.surfaces):
            is_air = surf.is_air
            ct = surf.thickness
            sd = surf.semi_diameter
            rad = surf.radius

            # 1. Diameter / Clear Aperture Dimension (Vertical or perpendicular)
            if sd > 0.05:
                p_top = self._transform_point(surf, float(sd), 0.0, plane=plane, offset=offset)
                p_bot = self._transform_point(surf, float(-sd), 0.0, plane=plane, offset=offset)
                
                # Add diameter dimension text / leader
                diam_val = 2.0 * sd
                label_txt = f"%%c{diam_val:.1f} mm"
                dim_offset = -6.0
                
                p_text = (p_top[0] + dim_offset, (p_top[1] + p_bot[1]) / 2.0)
                self._draw_dimension_line(msp, p_bot, p_top, p_text, label_txt, layer="DIMENSIONS_LINEAR")

            # 2. Thickness / Air Gap Distance Dimension (Along optical axis)
            if ct > 0.1 and i < len(self.stack.surfaces) - 1:
                next_surf = self.stack.surfaces[i + 1]
                p_v1 = self._transform_point(surf, 0.0, 0.0, plane=plane, offset=offset)
                p_v2 = self._transform_point(next_surf, 0.0, 0.0, plane=plane, offset=offset)

                tag = "Dist" if is_air else "Tc"
                dim_label = f"{tag}={ct:.2f} mm"
                offset_val = 14.0 if is_air else 8.0
                
                self._draw_dimension_line(msp, p_v1, p_v2, None, dim_label, offset_val=offset_val, layer="DIMENSIONS_LINEAR")

            # 3. Component Tilt / Angular Dimension Callouts
            tilt_angle = getattr(surf, 'tilt_x', 0.0)
            if abs(tilt_angle) > 0.05 and (surf.is_reflective() or surf.surface_type == SurfaceType.GRATING or "DMD" in (surf.name or "").upper()):
                p_vert = self._transform_point(surf, 0.0, 0.0, plane=plane, offset=offset)
                ang_lbl = f"Tilt θ={abs(tilt_angle):.2f}°"
                ang_pos = (p_vert[0] + 6.0, p_vert[1] + 12.0)
                msp.add_leader(
                    vertices=[p_vert, (p_vert[0] + 3.0, ang_pos[1]), ang_pos],
                    dxfattribs={"layer": "ANNOTATIONS"}
                )
                msp.add_text(
                    ang_lbl,
                    height=1.8,
                    dxfattribs={"layer": "ANNOTATIONS"}
                ).set_placement(ang_pos, align=TextEntityAlignment.MIDDLE_LEFT)

            # 4. Radius of Curvature Callouts (for curved surfaces)
            if abs(rad) > 1e-4:
                r_txt = f"R={rad:.1f} mm"
                p_vert = self._transform_point(surf, 0.0, 0.0, plane=plane, offset=offset)
                p_callout = (p_vert[0] - 8.0, p_vert[1] - 10.0 if rad > 0 else p_vert[1] + 10.0)
                
                msp.add_leader(
                    vertices=[p_vert, (p_vert[0] - 4.0, p_callout[1]), p_callout],
                    dxfattribs={"layer": "DIMENSIONS_RADIAL"}
                )
                msp.add_text(
                    r_txt,
                    height=1.8,
                    dxfattribs={"layer": "DIMENSIONS_RADIAL"}
                ).set_placement(p_callout, align=TextEntityAlignment.MIDDLE_RIGHT)

    def _draw_dimension_line(
        self,
        msp,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p_txt: Optional[Tuple[float, float]],
        text: str,
        offset_val: float = 0.0,
        layer: str = "DIMENSIONS_LINEAR"
    ):
        """Draws a true aligned dimension line with perpendicular witness lines, arrowheads, and centered text."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return

        if offset_val != 0.0:
            nx = -dy / dist * offset_val
            ny = dx / dist * offset_val
            p1_off = (p1[0] + nx, p1[1] + ny)
            p2_off = (p2[0] + nx, p2[1] + ny)
            
            # Extension / witness lines
            msp.add_line(p1, (p1[0] + nx * 1.15, p1[1] + ny * 1.15), dxfattribs={"layer": layer})
            msp.add_line(p2, (p2[0] + nx * 1.15, p2[1] + ny * 1.15), dxfattribs={"layer": layer})
        else:
            p1_off = p1
            p2_off = p2
            nx, ny = 0.0, 0.0

        # Dimension line between extension points
        msp.add_line(p1_off, p2_off, dxfattribs={"layer": layer})
        
        # Arrowhead markers
        self._add_arrowhead(msp, p1_off, p2_off, layer)
        self._add_arrowhead(msp, p2_off, p1_off, layer)

        # Dimension text placement
        if p_txt is None:
            mid_x = (p1_off[0] + p2_off[0]) / 2.0 + (nx / abs(offset_val) * 1.2 if offset_val != 0.0 else 0.0)
            mid_y = (p1_off[1] + p2_off[1]) / 2.0 + (ny / abs(offset_val) * 1.2 if offset_val != 0.0 else 1.2)
            p_txt = (mid_x, mid_y)
            
        msp.add_text(
            text,
            height=1.8,
            dxfattribs={"layer": layer}
        ).set_placement(p_txt, align=TextEntityAlignment.MIDDLE_CENTER)

    def _add_arrowhead(self, msp, p_tip: Tuple[float, float], p_from: Tuple[float, float], layer: str, size: float = 1.8):
        """Adds a CAD dimension arrowhead pointing to p_tip."""
        dx = p_tip[0] - p_from[0]
        dy = p_tip[1] - p_from[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        ux, uy = dx / dist, dy / dist
        vx, vy = -uy, ux

        p_base = (p_tip[0] - ux * size, p_tip[1] - uy * size)
        p_side1 = (p_base[0] + vx * (size * 0.35), p_base[1] + vy * (size * 0.35))
        p_side2 = (p_base[0] - vx * (size * 0.35), p_base[1] - vy * (size * 0.35))

        msp.add_lwpolyline([p_tip, p_side1, p_side2], close=True, dxfattribs={"layer": layer})

    def _draw_prescription_table(self, msp, bounds: Dict[str, float]):
        """Draws an optical prescription table below the 2D layout."""
        table_x = bounds["min_x"]
        table_y = bounds["min_y"] - 25.0
        row_height = 4.5
        col_widths = [12.0, 32.0, 22.0, 26.0, 26.0, 22.0, 26.0]
        headers = ["Surf", "Name", "Type", "Radius (mm)", "Thick (mm)", "Glass", "Semi-Diam (mm)"]

        # Draw Table Title
        msp.add_text(
            "OPTICAL PRESCRIPTION DATA TABLE [DIMENSIONS IN MM]",
            height=2.5,
            dxfattribs={"layer": "TITLE_BLOCK"}
        ).set_placement((table_x, table_y + 4.0), align=TextEntityAlignment.BOTTOM_LEFT)

        # Draw Header Row
        curr_x = table_x
        total_width = sum(col_widths)
        msp.add_line((table_x, table_y), (table_x + total_width, table_y), dxfattribs={"layer": "TITLE_BLOCK"})
        
        for c_idx, h in enumerate(headers):
            msp.add_text(
                h,
                height=1.8,
                dxfattribs={"layer": "TITLE_BLOCK"}
            ).set_placement((curr_x + 1.0, table_y - 3.2), align=TextEntityAlignment.BOTTOM_LEFT)
            curr_x += col_widths[c_idx]

        curr_y = table_y - row_height
        msp.add_line((table_x, curr_y), (table_x + total_width, curr_y), dxfattribs={"layer": "TITLE_BLOCK"})

        # Draw Data Rows
        for i, s in enumerate(self.stack.surfaces):
            curr_x = table_x
            stype = str(s.surface_type) if hasattr(s.surface_type, 'value') else s.surface_type
            vals = [
                f"S{i}",
                str(s.name)[:15],
                stype[:10],
                f"{s.radius:.2f}" if abs(s.radius) > 1e-4 else "Infinity",
                f"{s.thickness:.2f}",
                str(s.glass),
                f"{s.semi_diameter:.2f}"
            ]
            for c_idx, v in enumerate(vals):
                msp.add_text(
                    v,
                    height=1.6,
                    dxfattribs={"layer": "TITLE_BLOCK"}
                ).set_placement((curr_x + 1.0, curr_y - 3.2), align=TextEntityAlignment.BOTTOM_LEFT)
                curr_x += col_widths[c_idx]
            
            curr_y -= row_height
            msp.add_line((table_x, curr_y), (table_x + total_width, curr_y), dxfattribs={"layer": "TITLE_BLOCK"})

        # Vertical borders
        curr_x = table_x
        for w in col_widths:
            msp.add_line((curr_x, table_y), (curr_x, curr_y), dxfattribs={"layer": "TITLE_BLOCK"})
            curr_x += w
        msp.add_line((curr_x, table_y), (curr_x, curr_y), dxfattribs={"layer": "TITLE_BLOCK"})

    def _export_pure_python_dxf(self, file_path: str, include_dimensions: bool, include_rays: bool) -> str:
        """Pure-Python DXF R12 generator fallback if ezdxf is missing."""
        lines = [
            "0", "SECTION",
            "2", "ENTITIES"
        ]
        for i, surf in enumerate(self.stack.surfaces):
            p1 = self._transform_point(surf, float(surf.semi_diameter), 0.0)
            p2 = self._transform_point(surf, float(-surf.semi_diameter), 0.0)
            lines.extend([
                "0", "LINE",
                "8", "OPTICS",
                "10", f"{p1[0]:.4f}", "20", f"{p1[1]:.4f}", "30", "0.0",
                "11", f"{p2[0]:.4f}", "21", f"{p2[1]:.4f}", "31", "0.0"
            ])
        lines.extend(["0", "ENDSEC", "0", "EOF"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return file_path

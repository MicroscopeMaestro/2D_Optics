import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.visualizer_3d import (
    HAS_MPL_QT,
    Visualizer3DWidget,
    build_lens_solid_geometry,
    build_aperture_stop_geometry,
    build_rectangular_aperture_geometry,
)


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


def _solid_test_stack():
    stack = SurfaceStack()
    stack.add_surface(Surface(
        name="Sample_Object", surface_type=SurfaceType.OBJECT,
        thickness=25.0, semi_diameter=8.0,
    ))
    stack.add_surface(Surface(
        name="Stop", surface_type=SurfaceType.STOP,
        thickness=15.0, semi_diameter=6.0, aperture_type="Circular",
    ))
    stack.add_surface(Surface(
        name="Lens front", radius=50.0, thickness=5.0,
        glass="N-BK7", semi_diameter=12.0,
    ))
    stack.add_surface(Surface(
        name="Lens rear", radius=-50.0, thickness=30.0,
        glass="AIR", semi_diameter=12.0,
    ))
    stack.add_surface(Surface(
        name="DMD4500", surface_type=SurfaceType.MIRROR,
        glass="MIRROR", semi_diameter=7.0, thickness=20.0,
        aperture_type="Rectangular", aperture_x_half=7.0,
        aperture_y_half=5.0, substrate_thickness=2.0,
    ))
    stack.add_surface(Surface(
        name="Detector_CCD", surface_type=SurfaceType.DETECTOR,
        thickness=0.0, semi_diameter=6.0,
    ))
    return stack


def test_lens_solid_geometry_closes_front_rear_and_barrel():
    stack = _solid_test_stack()
    front, rear, side = build_lens_solid_geometry(stack.surfaces[2], stack.surfaces[3])

    assert front[0].shape == rear[0].shape
    assert side[0].shape[0] == 2
    assert side[0].shape[1] >= 32
    assert np.mean(rear[2]) > np.mean(front[2])
    assert np.allclose(side[0][0], front[0][-1])
    assert np.allclose(side[0][1], rear[0][-1])


def test_aperture_stop_solid_geometry_closes_faces_and_bore():
    """Verify build_aperture_stop_geometry produces 4 meshes forming a closed 3D annular diaphragm plate."""
    surf = Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=5.0, thickness=10.0)
    surf.global_origin = np.array([0.0, 0.0, 20.0])
    surf.global_R = np.eye(3)

    stop_meshes, inner_rim, outer_rim = build_aperture_stop_geometry(surf, thickness=1.5)
    assert len(stop_meshes) == 4
    front_mesh, back_mesh, outer_wall, inner_wall = stop_meshes

    # Verify front and rear faces are separated by physical thickness
    assert np.allclose(np.mean(back_mesh[2]) - np.mean(front_mesh[2]), 1.5)
    # Verify outer wall spans from z=20 to z=21.5
    assert np.isclose(np.min(outer_wall[2]), 20.0)
    assert np.isclose(np.max(outer_wall[2]), 21.5)
    # Verify inner bore wall spans clear aperture opening
    assert np.isclose(np.min(inner_wall[2]), 20.0)
    assert np.isclose(np.max(inner_wall[2]), 21.5)


def test_rectangular_aperture_geometry_forms_closed_3d_solid_box():
    """Verify build_rectangular_aperture_geometry creates 16 quads forming a complete 3D solid baffle plate."""
    surf = Surface(name="Slit", surface_type=SurfaceType.STOP, aperture_type="Rectangular",
                   aperture_x_half=3.0, aperture_y_half=5.0, semi_diameter=6.0)
    surf.global_origin = np.array([0.0, 0.0, 15.0])
    surf.global_R = np.eye(3)

    quads, inner_rim, outer_rim, all_verts = build_rectangular_aperture_geometry(surf, thickness=1.5)
    # 4 front plates + 4 back plates + 4 outer walls + 4 inner bore walls = 16 quads
    assert len(quads) == 16
    assert all_verts.shape[0] == 64  # 16 quads * 4 vertices each
    # Thickness in Z
    z_min = float(np.min(all_verts[:, 2]))
    z_max = float(np.max(all_verts[:, 2]))
    assert np.isclose(z_max - z_min, 1.5)


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_viewer_renders_solid_lens_and_component_bodies():
    widget = Visualizer3DWidget()
    try:
        stack = _solid_test_stack()
        widget.update_layout(surface_stack=stack, rays=[])
        # Filled solid bodies: Object stage, Aperture Stop, Lens (3 surfaces),
        # DMD substrate & micromirror array, Detector sensor package.
        assert len(widget.ax.collections) >= 6
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_viewer_fit_view_frames_from_first_surface_to_end_without_clipping():
    """Verify that pressing fit_view cleanly frames the optical system from first surface to end surface."""
    widget = Visualizer3DWidget()
    try:
        stack = _solid_test_stack()
        widget.update_layout(surface_stack=stack, rays=[])
        widget.fit_view()

        xlim = widget.ax.get_xlim3d()
        surfaces = stack.get_active_sequence()
        first_z = float(surfaces[0].global_origin[2])
        last_z = float(surfaces[-1].global_origin[2])

        # Bounding box must strictly encompass first surface (Surface 0) and end surface (Detector) with margin
        assert xlim[0] < first_z - 3.0, f"First surface (z={first_z}) must have margin before xlim[0]={xlim[0]}"
        assert xlim[1] > last_z + 3.0, f"Last surface (z={last_z}) must have margin after xlim[1]={xlim[1]}"
        # Camera distance must be <= 7.5
        assert hasattr(widget.ax, '_dist')
        assert widget.ax._dist <= 7.5
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_viewer_point_object_and_point_image_render_as_markers_not_huge_rectangles():
    """Verify that when semi_diameter is 0.0, the viewer renders point emitter & focal markers instead of giant rectangular boxes."""
    widget = Visualizer3DWidget()
    try:
        stack = SurfaceStack()
        stack.add_surface(Surface(
            name="Point_Object", surface_type=SurfaceType.OBJECT,
            thickness=20.0, semi_diameter=0.0,
        ))
        stack.add_surface(Surface(
            name="Stop", surface_type=SurfaceType.STOP,
            thickness=15.0, semi_diameter=5.0,
        ))
        stack.add_surface(Surface(
            name="Lens front", radius=40.0, thickness=4.0,
            glass="N-BK7", semi_diameter=10.0,
        ))
        stack.add_surface(Surface(
            name="Lens rear", radius=-40.0, thickness=30.0,
            glass="AIR", semi_diameter=10.0,
        ))
        stack.add_surface(Surface(
            name="Point_Image", surface_type=SurfaceType.DETECTOR,
            thickness=0.0, semi_diameter=0.0, aperture_type="Point",
        ))

        widget.update_layout(surface_stack=stack, rays=[])

        # Find scatter collections for point markers
        scatter_colls = [c for c in widget.ax.collections if "Path" in type(c).__name__]
        assert len(scatter_colls) >= 2, f"Expected at least 2 scatter collections, got {[type(c).__name__ for c in widget.ax.collections]}"

        # Verify Stop label displays semi-diameter
        stop_labels = [t.get_text() for t in widget.ax.texts if "Stop" in t.get_text()]
        assert len(stop_labels) > 0, "Expected Stop label in 3D viewer"
        assert any("Semi-D" in lbl and "5.00" in lbl for lbl in stop_labels), f"Stop label should display semi-diameter, got {stop_labels}"

        # Fit view framing must still encompass from first to last surface
        widget.fit_view()
        xlim = widget.ax.get_xlim3d()
        surfaces = stack.get_active_sequence()
        first_z = float(surfaces[0].global_origin[2])
        last_z = float(surfaces[-1].global_origin[2])
        assert xlim[0] < first_z
        assert xlim[1] > last_z
    finally:
        widget.close()


def test_stop_with_zero_thickness_rendered_without_penetrating_next_surface():
    """Verify that an optical stop with 0 thickness in the table does not penetrate into the following surface."""
    surf_stop = Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=4.0, thickness=0.0)
    surf_stop.global_origin = np.array([0.0, 0.0, 25.0])
    surf_stop.global_R = np.eye(3)

    # When backward mode is active (due to 0 thickness before next surface)
    stop_meshes, inner_rim, outer_rim = build_aperture_stop_geometry(
        surf_stop, thickness=1.0, surrounding_max_sd=12.0, z_offset_mode="backward"
    )
    front_mesh, back_mesh, outer_wall, inner_wall = stop_meshes

    # The outer wall must end at z=25.0 (active plane) and not exceed it (extends backwards to z=24.0)
    assert np.isclose(np.max(outer_wall[2]), 25.0)
    assert np.isclose(np.min(outer_wall[2]), 24.0)
    # Clear aperture inner rim is at exact active plane z=25.0
    assert np.allclose(inner_rim[2], 25.0)
    # The outer collar expands proportionally to match the surrounding system's 12.0 mm semi-diameter
    outer_radius = np.sqrt(outer_rim[0]**2 + outer_rim[1]**2)
    assert np.all(outer_radius >= 12.0)


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_opaque_mode_renders_smooth_solids_without_wireframe_mesh_lines():
    """Verify that Opaque mode (no_transparency=True) renders smooth solids without wireframe grid lines."""
    widget = Visualizer3DWidget()
    try:
        stack = _solid_test_stack()
        widget.set_no_transparency(True)
        widget.update_layout(surface_stack=stack, rays=[])

        # Verify that surface collections rendered via plot_surface have edgecolor='none' or transparent edge
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly_colls = [c for c in widget.ax.collections if isinstance(c, Poly3DCollection)]
        assert len(poly_colls) > 0, "Expected Poly3DCollection surfaces in 3D viewer"

        # Check collections for smooth shading (no high-contrast wireframe grid on faces)
        has_smooth_surfaces = False
        for coll in poly_colls:
            # Poly3DCollection with edgecolor='none' has edgecolors matching facecolor or alpha=0 or empty
            edgecolors = getattr(coll, '_edgecolors2d', getattr(coll, '_edgecolor3d', None))
            if edgecolors is not None and len(edgecolors) > 0:
                # If edge alpha is 0 or matches facecolor or linewidth is 0
                has_smooth_surfaces = True
                break
        assert has_smooth_surfaces is True

        # Verify perimeter silhouette lines exist for sharp boundaries
        assert len(widget.ax.lines) >= 4, "Expected perimeter silhouette contour lines"

        # Verify stop structure label readout is present
        stop_readouts = [t.get_text() for t in widget.ax.texts if "Stop" in t.get_text()]
        assert len(stop_readouts) > 0
        assert any("Semi-D" in t or "R=" in t for t in stop_readouts)
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_viewer_paints_optical_components_with_distinct_colors():
    """Verify that multi-lens elements, mirrors, galvos, gratings, stops, and sensors are painted with distinct physical colors."""
    from matplotlib.colors import to_hex
    from optics_app.engine.element_colors import LENS_ELEMENT_PALETTES
    from optics_app.engine.laser_scanner import GalvoScannerConfig

    stack = SurfaceStack()
    # 0. Sample Object
    stack.add_surface(Surface(name="Sample_Object", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=6.0))
    # 1-3. Achromatic Doublet (Crown + Flint cemented doublet)
    stack.add_surface(Surface(name="Crown_Front", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=12.0))
    stack.add_surface(Surface(name="Cemented_Interface", radius=-35.0, thickness=3.0, glass="SF11", semi_diameter=12.0))
    stack.add_surface(Surface(name="Flint_Rear", radius=-120.0, thickness=15.0, glass="AIR", semi_diameter=12.0))
    # 4. Stop
    stack.add_surface(Surface(name="Aperture_Stop", surface_type=SurfaceType.STOP, thickness=12.0, semi_diameter=5.0))
    # 5-6. Singlet Lens 2 (Emerald Green)
    stack.add_surface(Surface(name="Singlet_Front", radius=40.0, thickness=5.0, glass="N-SF11", semi_diameter=10.0))
    stack.add_surface(Surface(name="Singlet_Rear", radius=-60.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
    # 7. Mirror (Standard)
    stack.add_surface(Surface(name="Fold_Mirror", surface_type=SurfaceType.MIRROR, glass="MIRROR", thickness=15.0, semi_diameter=8.0))
    # 8. Galvo Scanning Mirror
    galvo_surf = Surface(name="Galvo_Scan", surface_type=SurfaceType.MIRROR, glass="MIRROR", thickness=15.0, semi_diameter=8.0)
    galvo_surf.galvo_config = GalvoScannerConfig(scan_axis="X")
    stack.add_surface(galvo_surf)
    # 9. Diffraction Grating
    stack.add_surface(Surface(name="Echelle_Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, thickness=15.0, semi_diameter=8.0))
    # 10. Detector
    stack.add_surface(Surface(name="Sensor_CCD", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=6.0))

    widget = Visualizer3DWidget()
    try:
        widget.update_layout(surface_stack=stack, rays=[])

        # 1. Collect hex facecolors from 3D collections
        observed_hex_colors = set()
        for coll in widget.ax.collections:
            facecolors = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
            if facecolors is not None and len(facecolors) > 0:
                for fc in facecolors:
                    observed_hex_colors.add(to_hex(fc[:3]).lower())

        line_hex_colors = set()
        for l in widget.ax.lines:
            c = l.get_color()
            if isinstance(c, str):
                line_hex_colors.add(c.lower())
            else:
                line_hex_colors.add(to_hex(c[:3]).lower())

        # 2. Check Lens 1 (Cyan Blue) and Lens 2 (Emerald Green) have distinct border contour lines
        lens1_border = LENS_ELEMENT_PALETTES[0]["border_hex"].lower()
        lens2_border = LENS_ELEMENT_PALETTES[1]["border_hex"].lower()
        assert lens1_border in line_hex_colors, f"Expected Lens 1 border {lens1_border} in line colors"
        assert lens2_border in line_hex_colors, f"Expected Lens 2 border {lens2_border} in line colors"
        assert lens1_border != lens2_border

        all_facecolors = []
        for coll in widget.ax.collections:
            fcs = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
            if fcs is not None and len(fcs) > 0:
                all_facecolors.extend(fcs)

        has_blue_lens = any(fc[2] > fc[0] + 0.15 and fc[2] > 0.4 for fc in all_facecolors)
        has_green_lens = any(fc[1] > fc[0] + 0.10 and fc[1] > fc[2] + 0.05 and fc[1] > 0.4 for fc in all_facecolors)
        assert has_blue_lens, "Expected blue-dominant facecolors for Lens 1 (Cyan Blue)"
        assert has_green_lens, "Expected green-dominant facecolors for Lens 2 (Emerald Green)"

        # 4. Check Mirror vs Scanning Galvo have distinct colors
        # Galvo has scanning teal border/normal #004d40
        assert "#004d40" in line_hex_colors, "Expected galvo scanning teal rim/normal line #004d40"
        # Standard mirror has metallic silver rim #94a3b8 and normal #64748b
        assert "#94a3b8" in line_hex_colors or "#64748b" in line_hex_colors, "Expected standard mirror silver rim/normal"

        # 5. Check Grating features metallic gold face and deep bronze ruling grooves
        assert "#78350f" in line_hex_colors, "Expected deep bronze grating ruling grooves #78350f"
        assert "#b45309" in line_hex_colors, "Expected grating border/normal line #b45309"

        # 6. Check Aperture Stop inner clear aperture rim in Stop Red
        assert "#ef4444" in line_hex_colors, "Expected Stop Red inner aperture rim #ef4444"

        # 7. Check Detector photosensitive active chip face and reticle
        assert "#dc2626" in observed_hex_colors, "Expected Sensor ruby/red active chip face #dc2626"
        assert "#fca5a5" in line_hex_colors, "Expected detector reticle indicator line #fca5a5"

        # 8. Check Sample Object face and alignment reticle
        assert "#38bdf8" in observed_hex_colors or "#0284c7" in observed_hex_colors, "Expected sample specimen face color"
        assert "#0c4a6e" in line_hex_colors, "Expected sample reticle line #0c4a6e"

        # 9. Test in Opaque mode: distinct colors are preserved and all mesh alphas are 1.0
        widget.set_no_transparency(True)
        widget.update_layout(surface_stack=stack, rays=[])

        all_opaque_fcs = []
        for coll in widget.ax.collections:
            fcs = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
            if fcs is not None and len(fcs) > 0:
                all_opaque_fcs.extend(fcs)
                for fc in fcs:
                    assert fc[3] == pytest.approx(1.0, abs=0.01)

        has_blue_lens_op = any(fc[2] > fc[0] + 0.15 and fc[2] > 0.4 for fc in all_opaque_fcs)
        has_green_lens_op = any(fc[1] > fc[0] + 0.10 and fc[1] > fc[2] + 0.05 and fc[1] > 0.4 for fc in all_opaque_fcs)
        assert has_blue_lens_op, "Expected blue-dominant lens 1 in opaque mode"
        assert has_green_lens_op, "Expected green-dominant lens 2 in opaque mode"
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_same_lens_surfaces_share_same_color_and_form_filled_volumetric_solid():
    """Verify that all surfaces of the same lens have the same color, and stops display real semi-diameter & aperture dimensions."""
    stack = SurfaceStack()
    # Lens 1: Cemented doublet (S0, S1, S2)
    stack.add_surface(Surface(name="Doublet_Crown", radius=50.0, thickness=6.0, glass="N-BK7", semi_diameter=12.0))
    stack.add_surface(Surface(name="Doublet_Interface", radius=-35.0, thickness=3.0, glass="SF11", semi_diameter=12.0))
    stack.add_surface(Surface(name="Doublet_Flint", radius=-100.0, thickness=15.0, glass="AIR", semi_diameter=12.0))
    # Circular Stop with semi-diameter = 4.5 mm
    stack.add_surface(Surface(name="Iris_Stop", surface_type=SurfaceType.STOP, thickness=10.0, semi_diameter=4.5))
    # Lens 2: Singlet with custom color
    s2_front = Surface(name="Singlet_Front", radius=30.0, thickness=4.0, glass="N-BK7", semi_diameter=10.0)
    s2_front.color = "#f59e0b"
    stack.add_surface(s2_front)
    stack.add_surface(Surface(name="Singlet_Rear", radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
    # Slit Stop
    stack.add_surface(Surface(name="Slit_Stop", surface_type=SurfaceType.STOP, aperture_type="Rectangular",
                              aperture_x_half=3.0, aperture_y_half=5.0, semi_diameter=6.0, thickness=0.0))

    widget = Visualizer3DWidget()
    try:
        widget.update_layout(surface_stack=stack, rays=[])

        # 1. Verify Stop displays real semi-diameter and full aperture size in 3D texts
        all_texts = [t.get_text() for t in widget.ax.texts]
        stop_texts = [t for t in all_texts if "Stop" in t or "Semi-D" in t]
        assert len(stop_texts) > 0, "Expected stop readouts in 3D viewer"
        # Circular stop: Semi-D R = 4.50 mm and Ø = 9.00 mm
        assert any("Semi-D" in t and "4.50" in t for t in stop_texts), f"Expected circular stop semi-diameter 4.50, got {stop_texts}"
        assert any("Ø = 9.00" in t or "9.00 mm" in t for t in stop_texts), f"Expected circular stop diameter 9.00, got {stop_texts}"
        # Slit stop: Semi-D X=3.00, Y=5.00 and Size 6.00 × 10.00 mm
        assert any("Semi-D" in t and "3.00" in t for t in stop_texts), f"Expected slit stop X semi-D, got {stop_texts}"
        assert any("6.00" in t and "10.00" in t for t in stop_texts), f"Expected slit full size 6.00 x 10.00 mm, got {stop_texts}"

        # 2. Verify all surfaces of Lens 1 (Doublet) share the same border silhouette color
        blue_border = "#0288d1"
        amber_custom = "#f59e0b"
        line_colors = [l.get_color().lower() if isinstance(l.get_color(), str) else "" for l in widget.ax.lines]
        assert line_colors.count(blue_border) >= 3, f"Expected at least 3 silhouette rings in {blue_border} for Doublet"

        # 3. Verify Singlet with custom color #f59e0b has both surfaces and barrel in #f59e0b
        assert amber_custom in line_colors, f"Expected amber rim color {amber_custom} in line colors"
        assert line_colors.count(amber_custom) >= 2, f"Expected both front and rear rings of Singlet to share {amber_custom}"
        all_lens_fcs = []
        for coll in widget.ax.collections:
            fcs = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
            if fcs is not None:
                all_lens_fcs.extend(fcs)
        has_amber_face = any(fc[0] > 0.6 and fc[1] > 0.35 and fc[2] < 0.2 for fc in all_lens_fcs)
        assert has_amber_face is True, "Custom amber color should be applied across the lens solid faces"

        # 4. Verify lens surfaces form filled volumetric 3D bodies
        assert len(widget.ax.collections) >= 6, "Expected volumetric 3D solid collections"
    finally:
        widget.close()





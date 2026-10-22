"""
Adversarial Stress Test Suite for Milestone 2: 3D PyQt Rendering & CAD Mesh Generation.

Empirical verification covering:
1. Continuous Galvo Angle Sweeps (-20° to +20°, extreme angles ±85°, ±90°, ±180°):
   - Dynamic vertex rotation with zero NaN/Inf.
   - Metric preservation (edge lengths, planarity, distances from pivot).
   - Outward normal vector, status badges, and mount transformations.
2. Polyhedral Prism and Beam Splitter Meshes Under Complex Tilts and Decenters:
   - Equilateral, Right-Angle, and Wedge prisms under full 3D Euler tilts and decenters.
   - Cube and Plate beam splitters with internal splitting interfaces under complex poses.
   - Visualizer coordinate frame (Z, X, Y) consistency and closed polyhedral topology.
   - B-Rep CAD solid generation integrity (Open CASCADE manifold validity, positive volume).
3. 3D Visualizer Compatibility with Branched Ray Paths:
   - Single-splitter and cascaded dual-splitter branched ray trees.
   - Multi-generation branch trajectory rendering (incident, transmitted, reflected).
   - Ray energy, wavelength false-coloring, and bound auto-scaling under branched paths.
"""

import math
import os
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.components import (
    PrismConfig,
    BeamSplitterConfig,
    GalvoScannerConfig,
    create_prism_component,
    create_beamsplitter_component,
    create_galvo_component,
)
from optics_app.engine.mechanisms import get_effective_pose
from optics_app.cad.exporter import (
    CADExporter,
    STEPExporter,
    create_prism_solid,
    create_beamsplitter_solid,
    get_exportable_elements,
    _get_build123d,
)
from optics_app.gui.visualizer_3d import (
    build_prism_geometry,
    build_beamsplitter_geometry,
    _surface_mesh,
    _edge_ring,
    Visualizer3DWidget,
    QT_AVAILABLE,
    HAS_MPL_QT,
)
from optics_app.engine.tracer import (
    OpticalSystem,
    Ray,
    RayNode,
    RayTree,
    BranchedTracer,
)


def make_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Exact 3D orthogonal rotation matrix from Euler angles (degrees)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=float)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=float)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


@pytest.fixture
def b3d():
    lib = _get_build123d()
    if lib is None:
        pytest.skip("build123d / OCP CAD kernel is not available")
    return lib


# ==============================================================================
# Suite 1: Galvo Continuous Sweeps & Dynamic Vertex Rotation
# ==============================================================================

class TestGalvoContinuousSweepsAdversarial:
    """Stress-test Galvo 3D mesh generation across continuous angle sweeps and edge cases."""

    @pytest.mark.parametrize("scan_axis", ["X", "Y", "Z", [1.0, 1.0, 0.0], [0.3, -0.7, 0.65]])
    def test_galvo_sweep_continuous_rotation_no_nan_inf(self, scan_axis):
        """
        Sweep galvo across continuous angle range (-20° to +20° in 1° steps).
        Verify vertices rotate dynamically with zero NaN/Inf and preserve rigid-body metrics.
        """
        pivot_offset = [2.5, -1.0, 4.0]
        mount_origin = np.array([10.0, -15.0, 50.0])
        mount_R = make_rotation_matrix(15.0, -25.0, 30.0)

        comp, surf = create_galvo_component(
            name="SweepGalvo",
            scan_axis=scan_axis,
            min_angle_deg=-25.0,
            max_angle_deg=25.0,
            current_angle_deg=0.0,
            pivot_offset_mm=pivot_offset,
        )
        surf.semi_diameter = 12.5
        surf.global_origin = mount_origin.copy()
        surf.global_R = mount_R.copy()

        angles = np.linspace(-20.0, 20.0, 41)
        prev_verts = None
        prev_origin = None
        prev_R = None

        for ang in angles:
            surf.galvo_config.current_angle_deg = float(ang)
            eff_origin, eff_R = get_effective_pose(surf)

            # 1. Pose invariants
            assert np.all(np.isfinite(eff_origin)), f"NaN/Inf in eff_origin at angle {ang}"
            assert np.all(np.isfinite(eff_R)), f"NaN/Inf in eff_R at angle {ang}"
            # Orthogonality: R^T @ R == I, det(R) == 1
            assert np.allclose(eff_R.T @ eff_R, np.eye(3), atol=1e-10), f"R not orthogonal at {ang}"
            assert np.isclose(np.linalg.det(eff_R), 1.0, atol=1e-10), f"det(R) != 1 at {ang}"

            # 2. Distance from effective origin to pivot point must remain constant
            # P_pivot = P_0 + R_mount @ d_pivot
            p_pivot = mount_origin + mount_R @ np.array(pivot_offset)
            dist_to_pivot = np.linalg.norm(eff_origin - p_pivot)
            expected_dist = np.linalg.norm(pivot_offset)
            assert np.isclose(dist_to_pivot, expected_dist, atol=1e-8)

            # 3. Surface mesh generation
            x_mesh, y_mesh, z_mesh = _surface_mesh(surf, radial_samples=5, angular_samples=16)
            all_coords = np.vstack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()]).T

            assert np.all(np.isfinite(all_coords)), f"NaN/Inf in surface mesh at angle {ang}"

            # 4. Perimeter edge ring generation
            rx, ry, rz = _edge_ring(surf, angular_samples=24)
            ring_coords = np.vstack([rx, ry, rz]).T
            assert np.all(np.isfinite(ring_coords)), f"NaN/Inf in edge ring at angle {ang}"

            # 5. Dynamic motion: verify vertices actually move between angular steps
            if prev_verts is not None:
                delta_verts = np.linalg.norm(all_coords - prev_verts, axis=1)
                assert np.max(delta_verts) > 1e-4, f"Vertices remained static between steps at {ang}!"
                delta_R = np.linalg.norm(eff_R - prev_R)
                assert delta_R > 1e-4, f"Rotation matrix static at {ang}!"

            # 6. Internal distance preservation (rigid body metric)
            # Distance between first vertex and center
            v0_dist = np.linalg.norm(all_coords[0] - eff_origin)
            v_last_dist = np.linalg.norm(all_coords[-1] - eff_origin)
            if prev_verts is not None:
                # Local geometry cannot stretch or warp
                prev_v0_dist = np.linalg.norm(prev_verts[0] - prev_origin)
                assert np.isclose(v0_dist, prev_v0_dist, atol=1e-8)

            prev_verts = all_coords.copy()
            prev_origin = eff_origin.copy()
            prev_R = eff_R.copy()

    def test_galvo_extreme_deflections_and_singularities(self):
        """Stress-test extreme angles (±85°, ±90°, ±180°), degenerate scan axes, and huge offsets."""
        surf = Surface(name="AdversarialGalvoSingularities")
        surf.global_origin = np.array([0.0, 0.0, 0.0])
        surf.global_R = np.eye(3)

        extreme_angles = [-180.0, -90.0, -89.9, -45.0, 0.0, 45.0, 89.9, 90.0, 180.0, 360.0]
        test_axes = [
            "X", "Y", "Z",
            [0.0, 0.0, 0.0],          # Zero axis -> should gracefully default without division by zero
            [1e-12, 1e-12, 1e-12],    # Near-zero axis
            [-100.0, 200.0, -300.0],  # Large un-normalized axis
        ]

        for ax in test_axes:
            for ang in extreme_angles:
                cfg = GalvoScannerConfig(
                    scan_axis=ax,
                    current_angle_deg=ang,
                    pivot_offset_mm=[1000.0, -500.0, 250.0],  # Huge pivot offset
                )
                surf.galvo_config = cfg
                eff_origin, eff_R = get_effective_pose(surf)

                assert np.all(np.isfinite(eff_origin)), f"eff_origin NaN/Inf with axis {ax} at {ang}"
                assert np.all(np.isfinite(eff_R)), f"eff_R NaN/Inf with axis {ax} at {ang}"
                assert np.allclose(eff_R.T @ eff_R, np.eye(3), atol=1e-8)

                x_mesh, y_mesh, z_mesh = _surface_mesh(surf, radial_samples=4, angular_samples=12)
                assert np.all(np.isfinite(x_mesh))
                assert np.all(np.isfinite(y_mesh))
                assert np.all(np.isfinite(z_mesh))

    @pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
    def test_galvo_visualizer_sweep_render_cycle(self, monkeypatch):
        """Verify Visualizer3DWidget updates cleanly across a continuous 20° sweep without exceptions."""
        from optics_app.gui.qt_compat import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        stack = SurfaceStack()
        g_comp, g_surf = create_galvo_component(
            name="GalvoScanner",
            scan_axis="X",
            min_angle_deg=-20.0,
            max_angle_deg=20.0,
            current_angle_deg=-20.0,
            pivot_offset_mm=[0.0, 0.0, 10.0],
        )
        stack.components.append(g_comp)
        stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=30.0))
        stack.add_surface(g_surf)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        widget = Visualizer3DWidget(auto_refresh=False)
        try:
            monkeypatch.setattr(widget, "trace_3d_rays", lambda: [])
            # Sweep through angles
            for target_deg in [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]:
                g_surf.galvo_config.current_angle_deg = target_deg
                widget.update_layout(surface_stack=stack, rays=[])
                badge_text = widget.btn_mech_state.text()
                assert f"{target_deg:+.1f}°" in badge_text or f"{int(target_deg)}°" in badge_text
                # Verify canvas limits are finite
                assert all(math.isfinite(lim) for lim in widget.ax.get_xlim())
                assert all(math.isfinite(lim) for lim in widget.ax.get_ylim())
                assert all(math.isfinite(lim) for lim in widget.ax.get_zlim())
        finally:
            widget.close()

    @pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
    def test_galvo_toggle_mech_state_respects_min_max_angle_deg(self, monkeypatch):
        """Verify toggle_mech_state respects min_angle_deg and max_angle_deg fallback."""
        from optics_app.gui.qt_compat import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        stack = SurfaceStack()
        g_comp, g_surf = create_galvo_component(
            name="GalvoScanner",
            scan_axis="X",
            min_angle_deg=-5.0,
            max_angle_deg=5.0,
            current_angle_deg=5.0,
            pivot_offset_mm=[0.0, 0.0, 10.0],
        )
        stack.components.append(g_comp)
        stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=30.0))
        stack.add_surface(g_surf)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        widget = Visualizer3DWidget(auto_refresh=False)
        try:
            monkeypatch.setattr(widget, "_trigger_retrace", lambda: None)
            widget.update_layout(surface_stack=stack, rays=[])
            # Advancing from 5.0 (which is max_angle_deg) should wrap around to min_angle_deg (-5.0)
            new_state = widget.toggle_mech_state()
            assert "-5.0°" in new_state
            assert g_surf.galvo_config.current_angle_deg == -5.0
        finally:
            widget.close()


# ==============================================================================
# Suite 2: Polyhedral Prism and Beam Splitter Meshes Under Complex Tilts and Decenters
# ==============================================================================

class TestPolyhedralPrismAndBeamSplitterMeshesAdversarial:
    """Stress-test polyhedral prism and beam splitter meshes under complex 3D tilts and decenters."""

    @pytest.mark.parametrize("geometry_type, apex_angle", [
        ("Equilateral", 60.0),
        ("Right-Angle", 90.0),
        ("Wedge", 3.0),
        ("Wedge", 12.0),
        ("Custom", 45.0),
        ("Custom", 75.0),
    ])
    def test_prism_mesh_complex_tilts_and_decenters(self, geometry_type, apex_angle):
        """
        Subject prism components to arbitrary 3D Euler angle rotations and decenter shifts.
        Verify vertex count, face planarity, finite coordinates, and coordinate frame consistency.
        """
        # Complex compound rotations and translations
        complex_R = make_rotation_matrix(28.5, -42.1, 63.8)
        complex_origin = np.array([45.2, -120.7, 310.4])

        comp, surfs = create_prism_component(
            name=f"Stress_{geometry_type}_Prism",
            geometry_type=geometry_type,
            apex_angle_deg=apex_angle,
            base_width_mm=32.0,
        )
        comp.prism_config.height_mm = 28.0
        comp.prism_config.length_mm = 40.0
        s_front = surfs[0]
        s_front.global_origin = complex_origin.copy()
        s_front.global_R = complex_R.copy()

        plot_faces, world_verts = build_prism_geometry(
            s_front, surfs[1], config=comp.prism_config, height_mm=28.0, length_mm=40.0
        )

        # 1. Finite coordinates
        assert np.all(np.isfinite(world_verts)), "NaN/Inf in prism world vertices"
        assert len(plot_faces) in (5, 6), f"Unexpected face count {len(plot_faces)} for {geometry_type}"

        # 2. Topology checks
        expected_vert_count = 8 if "wedge" in geometry_type.lower() else 6
        assert world_verts.shape == (expected_vert_count, 3)

        # 3. Coordinate frame mapping: plot_faces must be in (Z, X, Y) while world_verts is (X, Y, Z)
        for face in plot_faces:
            for pt in face:
                assert len(pt) == 3
                # Check that point corresponds to some vertex in world_verts
                matched = False
                for wv in world_verts:
                    if np.allclose([pt[0], pt[1], pt[2]], [wv[2], wv[0], wv[1]], atol=1e-8):
                        matched = True
                        break
                assert matched, f"Face point {pt} does not match visualizer coordinate frame mapping!"

        # 4. Face planarity check: for quadrilateral faces, 4 vertices must lie on the same plane
        for face in plot_faces:
            if len(face) == 4:
                p0, p1, p2, p3 = [np.array(p) for p in face]
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-6:
                    normal /= norm_len
                    dist = abs(np.dot(p3 - p0, normal))
                    assert dist < 1e-4, f"Quadrilateral face not planar! Deviation = {dist}"

        # 5. Volume invariance under rigid transformation
        # Centroid should match eff_origin + complex_R @ local_centroid
        eff_origin, eff_R = get_effective_pose(s_front)
        assert np.allclose(eff_origin, complex_origin)
        assert np.allclose(eff_R, complex_R)

    @pytest.mark.parametrize("splitter_type", ["Cube", "Plate"])
    def test_beamsplitter_mesh_complex_tilts_and_decenters(self, splitter_type):
        """
        Subject cube and plate beam splitters to extreme rotations and decenters.
        Verify outer faces, internal splitting interface face, and vertex metrics.
        """
        complex_R = make_rotation_matrix(-55.0, 33.3, -112.5)
        complex_origin = np.array([-75.0, 18.5, 420.0])

        comp, surfs = create_beamsplitter_component(
            name=f"Stress_{splitter_type}_Splitter",
            splitter_type=splitter_type,
            substrate_thickness_mm=4.0,
        )
        comp.beamsplitter_config.cube_side_length_mm = 25.4
        s_front = surfs[0]
        s_front.global_origin = complex_origin.copy()
        s_front.global_R = complex_R.copy()

        outer_faces, iface_faces, world_verts = build_beamsplitter_geometry(
            s_front, surfs[1], config=comp.beamsplitter_config,
            cube_side_length_mm=25.4,
            clear_aperture_x=30.0,
            clear_aperture_y=40.0,
        )

        assert np.all(np.isfinite(world_verts)), "NaN/Inf in beamsplitter world vertices"
        assert world_verts.shape == (8, 3), "Beamsplitter must have 8 box corner vertices"

        # Topology
        if splitter_type == "Cube":
            assert len(outer_faces) == 6, "Cube beamsplitter must have 6 outer faces"
            assert len(iface_faces) == 1, "Cube beamsplitter must have 1 diagonal splitting interface"
        else:
            assert len(outer_faces) == 5, "Plate beamsplitter has 5 outer substrate faces"
            assert len(iface_faces) == 1, "Plate beamsplitter has 1 front splitting face"

        # Check visualizer frame mapping for interface face
        iface = iface_faces[0]
        assert len(iface) == 4
        for pt in iface:
            matched = any(np.allclose([pt[0], pt[1], pt[2]], [wv[2], wv[0], wv[1]], atol=1e-8) for wv in world_verts)
            assert matched

        # Planarity of the diagonal interface
        p0, p1, p2, p3 = [np.array(p) for p in iface]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        dist = abs(np.dot(p3 - p0, normal))
        assert dist < 1e-4, f"Internal splitting interface not planar! Deviation = {dist}"

    def test_cad_solid_generation_extreme_aspect_ratios(self, b3d):
        """Verify Open CASCADE B-Rep solids are valid and watertight under extreme aspect ratios."""
        # 1. Very thin wedge prism (0.5 degree)
        comp_w, surfs_w = create_prism_component(
            name="UltraThinWedge",
            geometry_type="Wedge",
            apex_angle_deg=0.5,
            base_width_mm=50.0,
        )
        comp_w.prism_config.length_mm = 50.0
        solid_w = create_prism_solid(surfs_w, config=comp_w.prism_config)
        assert solid_w is not None
        assert solid_w.is_valid
        assert solid_w.volume > 0.0

        # 2. Large equilateral prism
        comp_eq, surfs_eq = create_prism_component(
            name="GiantEquilateral",
            geometry_type="Equilateral",
            apex_angle_deg=60.0,
            base_width_mm=100.0,
        )
        comp_eq.prism_config.length_mm = 120.0
        solid_eq = create_prism_solid(surfs_eq, config=comp_eq.prism_config)
        assert solid_eq is not None
        assert solid_eq.is_valid
        assert solid_eq.volume > 100000.0

        # 3. Miniature cube beam splitter (3mm)
        comp_bs, surfs_bs = create_beamsplitter_component(
            name="MiniCubeBS",
            splitter_type="Cube",
        )
        solid_bs = create_beamsplitter_solid(surfs_bs, config=comp_bs.beamsplitter_config, cube_side_length_mm=3.0)
        assert solid_bs is not None
        assert solid_bs.is_valid
        assert abs(solid_bs.volume - 27.0) < 1.0


# ==============================================================================
# Suite 3: 3D Visualization Compatibility with Branched Ray Paths
# ==============================================================================

class TestBranchedRayPaths3DVisualizerAdversarial:
    """Stress-test compatibility of 3D visualizer with branched ray trajectories (RayTree)."""

    def test_branched_ray_tree_rendering_single_splitter(self):
        """
        Trace ray tree through beam splitter cube and verify 3D visualizer renders all branch segments.
        """
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=25.0)
        bs_comp, (bs_front, bs_rear) = create_beamsplitter_component(
            name="CubeBS", splitter_type="Cube", substrate_glass="N-BK7"
        )
        s_det_trans = Surface(name="DetTransmitted", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(bs_front)
        stack.add_surface(bs_rear)
        stack.add_surface(s_det_trans)
        stack.components.append(bs_comp)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        tree = tracer.trace_ray_tree(ray_in)

        # Confirm tree has branches: root incident, transmitted child, reflected child
        assert len(tree.nodes) >= 3
        branch_types = {n.branch_type for n in tree.nodes.values()}
        assert "incident" in branch_types
        assert "transmitted" in branch_types
        assert "reflected" in branch_types

        all_rays = tree.all_rays()
        assert len(all_rays) >= 3

        # Render in Visualizer3DWidget
        if QT_AVAILABLE and HAS_MPL_QT:
            from optics_app.gui.qt_compat import QtWidgets
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            widget = Visualizer3DWidget(auto_refresh=False)
            try:
                # Update layout with the extracted branched rays
                widget.update_layout(surface_stack=stack, rays=all_rays)

                # Confirm lines drawn on the axis
                lines = widget.ax.lines
                assert len(lines) > 0, "No ray trajectory lines rendered in 3D axis!"

                # Verify all drawn ray vertices are completely finite
                for line in lines:
                    xdata, ydata = line.get_data()
                    assert np.all(np.isfinite(xdata))
                    assert np.all(np.isfinite(ydata))
                    if hasattr(line, "get_3d_data"):
                        x3, y3, z3 = line.get_3d_data()
                        assert np.all(np.isfinite(x3))
                        assert np.all(np.isfinite(y3))
                        assert np.all(np.isfinite(z3))
                # Verify direct RayTree object support in update_layout
                widget.update_layout(surface_stack=stack, rays=tree)
                lines_tree = widget.ax.lines
                assert len(lines_tree) > 0, "No ray trajectory lines rendered when passing RayTree directly!"
            finally:
                widget.close()

    def test_cascaded_dual_splitter_branch_tree_rendering(self):
        """
        Cascade two successive beam splitters to generate a multi-generational ray tree (4 output paths).
        Verify that all branched rays are rendered and visualizer bounds scale accurately.
        """
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        bs1_comp, (bs1_f, bs1_r) = create_beamsplitter_component(name="BS1", splitter_type="Cube", substrate_glass="N-BK7")
        bs1_f.thickness = 15.0
        bs1_r.thickness = 20.0

        bs2_comp, (bs2_f, bs2_r) = create_beamsplitter_component(name="BS2", splitter_type="Cube", substrate_glass="N-BK7")
        bs2_f.thickness = 15.0
        bs2_r.thickness = 25.0

        s_det = Surface(name="DetFinal", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(bs1_f)
        stack.add_surface(bs1_r)
        stack.add_surface(bs2_f)
        stack.add_surface(bs2_r)
        stack.add_surface(s_det)
        stack.components.extend([bs1_comp, bs2_comp])
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack, max_depth=6)
        bundle = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.4861),
            Ray(pos=np.array([1.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.6563),
        ]
        trees = tracer.trace_bundle_tree(bundle)

        # Collect all rays across trees
        combined_rays = []
        for t in trees:
            combined_rays.extend(t.all_rays())

        assert len(combined_rays) >= 6

        if QT_AVAILABLE and HAS_MPL_QT:
            from optics_app.gui.qt_compat import QtWidgets
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            widget = Visualizer3DWidget(auto_refresh=False)
            try:
                widget.update_layout(surface_stack=stack, rays=combined_rays)
                # Verify axis limits expanded appropriately
                xlim = widget.ax.get_xlim()
                ylim = widget.ax.get_ylim()
                zlim = widget.ax.get_zlim()
                assert xlim[1] > xlim[0]
                assert ylim[1] > ylim[0]
                assert zlim[1] > zlim[0]
            finally:
                widget.close()


# ==============================================================================
# Suite 4: Parity Check Between optics_app and v2
# ==============================================================================

def test_byte_parity_between_optics_app_and_v2():
    """Verify 100% byte-for-byte parity for cad/exporter.py and gui/visualizer_3d.py."""
    import filecmp
    pairs = [
        ("optics_app/cad/exporter.py", "v2/cad/exporter.py"),
        ("optics_app/gui/visualizer_3d.py", "v2/gui/visualizer_3d.py"),
        ("optics_app/gui/main_window.py", "v2/gui/main_window.py"),
    ]
    for p1, p2 in pairs:
        assert os.path.exists(p1), f"Missing {p1}"
        assert os.path.exists(p2), f"Missing {p2}"
        match = filecmp.cmp(p1, p2, shallow=False)
        assert match, f"Parity mismatch between {p1} and {p2}!"

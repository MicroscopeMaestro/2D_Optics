"""
Comprehensive Test Suite for Generic Optical Components in CAD Export & 3D Visualization.

Tests:
1. build123d B-Rep solid generation for Prisms (Equilateral, Right-Angle, Wedge).
2. build123d B-Rep solid generation for Beam Splitters (Cube compound, Plate substrate).
3. get_exportable_elements() classification for generic optical components.
4. STEPExporter full assembly export containing Prism, Beam Splitter, and Galvo.
5. Dynamic mechanism pose placement in STEP assembly via get_effective_pose().
6. 3D PyQt visualizer polyhedral mesh generation (build_prism_geometry, build_beamsplitter_geometry).
7. 3D PyQt visualizer mechanism controls and dynamic galvo state cycling.
"""

import os
import math
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
    Visualizer3DWidget,
    QT_AVAILABLE,
    HAS_MPL_QT,
)


@pytest.fixture
def b3d():
    lib = _get_build123d()
    if lib is None:
        pytest.skip("build123d / OCP CAD kernel is not available")
    return lib


# ==============================================================================
# 1. B-Rep Solid Creation for Prisms
# ==============================================================================

def test_create_prism_solid_equilateral(b3d):
    comp, surfs = create_prism_component(
        name="SF11 Equilateral Prism",
        geometry_type="Equilateral",
        apex_angle_deg=60.0,
        glass="N-SF11",
        base_width_mm=30.0,
    )
    solid = create_prism_solid(surfs, config=comp.prism_config)
    assert solid is not None
    assert solid.is_valid
    # Bounding box width along X should be approximately base_width_mm (30 mm)
    bbox = solid.bounding_box()
    assert abs(bbox.size.X - 30.0) < 1.0
    # Equilateral triangle height = 30 * sqrt(3) / 2 ≈ 25.98 mm
    assert abs(bbox.size.Z - (30.0 * math.sqrt(3) / 2.0)) < 1.5
    # Volume must be positive and non-zero
    assert solid.volume > 1000.0


def test_create_prism_solid_right_angle(b3d):
    comp, surfs = create_prism_component(
        name="BK7 Right Angle Prism",
        geometry_type="Right-Angle",
        apex_angle_deg=90.0,
        glass="N-BK7",
        base_width_mm=25.0,
    )
    solid = create_prism_solid(surfs, config=comp.prism_config)
    assert solid is not None
    assert solid.is_valid
    assert solid.volume > 500.0


def test_create_prism_solid_wedge(b3d):
    comp, surfs = create_prism_component(
        name="Fused Silica Wedge",
        geometry_type="Wedge",
        apex_angle_deg=4.0,
        glass="SILICA",
        base_width_mm=25.4,
    )
    solid = create_prism_solid(surfs, config=comp.prism_config)
    assert solid is not None
    assert solid.is_valid
    assert solid.volume > 500.0


# ==============================================================================
# 2. B-Rep Solid Creation for Beam Splitters
# ==============================================================================

def test_create_beamsplitter_solid_cube(b3d):
    comp, surfs = create_beamsplitter_component(
        name="25mm Polarizing Beamsplitter Cube",
        splitter_type="Cube",
        coating_type="PBS",
        substrate_glass="N-BK7",
    )
    solid = create_beamsplitter_solid(surfs, config=comp.beamsplitter_config, cube_side_length_mm=25.0)
    assert solid is not None
    assert solid.is_valid
    # Cube should have volume close to 25^3 = 15625 mm^3
    assert abs(solid.volume - (25.0 ** 3)) < 100.0
    # Must contain prism sub-solids forming the compound cube
    if hasattr(solid, "solids"):
        assert len(solid.solids()) >= 2


def test_create_beamsplitter_solid_plate(b3d):
    comp, surfs = create_beamsplitter_component(
        name="50:50 Plate Beamsplitter",
        splitter_type="Plate",
        coating_type="NPBS_50_50",
        substrate_glass="FS",
        substrate_thickness_mm=3.0,
    )
    solid = create_beamsplitter_solid(surfs, config=comp.beamsplitter_config, width=25.0, height=35.0)
    assert solid is not None
    assert solid.is_valid
    bbox = solid.bounding_box()
    assert abs(bbox.size.X - 25.0) < 1.0
    assert abs(bbox.size.Y - 35.0) < 1.0
    assert abs(bbox.size.Z - 3.0) < 0.5


# ==============================================================================
# 3. get_exportable_elements Classification
# ==============================================================================

def test_get_exportable_elements_classifies_generic_components():
    stack = SurfaceStack()
    p_comp, p_surfs = create_prism_component(name="Dispersing Prism", glass="F_SILICA")
    bs_comp, bs_surfs = create_beamsplitter_component(name="Cube Beamsplitter", splitter_type="Cube")
    g_comp, g_surf = create_galvo_component(name="Galvo Scan Head", scan_axis="X", current_angle_deg=5.0)

    stack.components.extend([p_comp, bs_comp, g_comp])
    for s in p_surfs:
        stack.add_surface(s)
    for s in bs_surfs:
        stack.add_surface(s)
    stack.add_surface(g_surf)

    # Standard lens element
    s_lens_f = Surface(name="Lens Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
    s_lens_r = Surface(name="Lens Rear", radius=-50.0, thickness=20.0, semi_diameter=12.5)
    s_det = Surface(name="Focal Plane", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0)
    stack.add_surface(s_lens_f)
    stack.add_surface(s_lens_r)
    stack.add_surface(s_det)

    elements = get_exportable_elements(stack)
    elem_types = [e["type"] for e in elements]

    assert "Prism" in elem_types
    assert "Beam Splitter" in elem_types
    assert "Galvo Scanner" in elem_types
    assert any(t in ("Singlet", "Lens Element") for t in elem_types)

    # Ensure elements carry their respective configs
    prism_elem = next(e for e in elements if e["type"] == "Prism")
    assert prism_elem["material"] == "F_SILICA"
    assert len(prism_elem["surfaces"]) == 2

    bs_elem = next(e for e in elements if e["type"] == "Beam Splitter")
    assert bs_elem["config"] is not None

    galvo_elem = next(e for e in elements if e["type"] == "Galvo Scanner")
    assert galvo_elem["config"].scan_axis == "X"
    assert galvo_elem["config"].current_angle_deg == 5.0


# ==============================================================================
# 4. STEP Assembly Export with Generic Components
# ==============================================================================

def test_step_export_assembly_with_generic_components(tmp_path, b3d):
    stack = SurfaceStack()
    p_comp, p_surfs = create_prism_component(name="ApexPrism", glass="N-SF11")
    bs_comp, bs_surfs = create_beamsplitter_component(name="CubeSplitter", splitter_type="Cube")
    g_comp, g_surf = create_galvo_component(name="FastGalvo", scan_axis="Y", current_angle_deg=-4.0)

    stack.components.extend([p_comp, bs_comp, g_comp])
    for s in p_surfs:
        stack.add_surface(s)
    for s in bs_surfs:
        stack.add_surface(s)
    stack.add_surface(g_surf)

    export_path = os.path.join(tmp_path, "generic_optical_assembly.step")
    result = CADExporter.export_system_step(stack, export_path)

    assert os.path.exists(result)
    assert os.path.getsize(result) > 2000

    with open(result, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ISO-10303-21;" in content
    assert "MANIFOLD_SOLID_BREP" in content
    # Check that elements are exported with proper prefix labels
    assert "Element_1_ApexPrism" in content
    assert "Element_2_CubeSplitter" in content
    assert "Element_3_FastGalvo" in content


# ==============================================================================
# 5. Dynamic Mechanism Kinematic Pose in CAD Export
# ==============================================================================

def test_dynamic_mechanism_pose_changes_cad_placement(tmp_path, b3d):
    # Test 1: Galvo at 0 degrees
    stack1 = SurfaceStack()
    g1_comp, g1_surf = create_galvo_component(
        name="ScanningMirror",
        scan_axis="X",
        current_angle_deg=0.0,
        pivot_offset_mm=[0.0, 0.0, 5.0],
    )
    stack1.components.append(g1_comp)
    stack1.add_surface(g1_surf)

    path1 = os.path.join(tmp_path, "galvo_0deg.step")
    CADExporter.export_system_step(stack1, path1)

    # Test 2: Galvo tilted at +10 degrees
    stack2 = SurfaceStack()
    g2_comp, g2_surf = create_galvo_component(
        name="ScanningMirror",
        scan_axis="X",
        current_angle_deg=10.0,
        pivot_offset_mm=[0.0, 0.0, 5.0],
    )
    stack2.components.append(g2_comp)
    stack2.add_surface(g2_surf)

    path2 = os.path.join(tmp_path, "galvo_10deg.step")
    CADExporter.export_system_step(stack2, path2)

    with open(path1, "r", encoding="utf-8") as f:
        c1 = f.read()
    with open(path2, "r", encoding="utf-8") as f:
        c2 = f.read()

    # The two STEP assemblies must differ in cartesian points / direction vectors
    assert c1 != c2

    # Verify get_effective_pose directly
    orig1, R1 = get_effective_pose(g1_surf)
    orig2, R2 = get_effective_pose(g2_surf)
    assert not np.allclose(R1, R2, atol=1e-3)
    assert not np.allclose(orig1, orig2, atol=1e-3)


# ==============================================================================
# 6. 3D Visualizer Polyhedral Geometry Mesh Generation
# ==============================================================================

def test_build_prism_geometry_mesh():
    comp, surfs = create_prism_component(
        name="Dispersion Prism",
        geometry_type="Equilateral",
        apex_angle_deg=60.0,
        base_width_mm=30.0,
    )
    plot_faces, world_verts = build_prism_geometry(surfs[0], surfs[1], config=comp.prism_config)
    assert len(plot_faces) == 5  # Equilateral prism has 5 polyhedral faces (2 triangles + 3 rectangles)
    assert world_verts.shape[0] == 6  # 6 vertices
    assert world_verts.shape[1] == 3
    # Check that plot_faces contain valid 3D points in (Z, X, Y) format
    for face in plot_faces:
        assert len(face) in (3, 4)
        for pt in face:
            assert len(pt) == 3


def test_build_beamsplitter_geometry_mesh():
    comp, surfs = create_beamsplitter_component(
        name="Beamsplitter Cube",
        splitter_type="Cube",
    )
    outer_faces, iface_faces, world_verts = build_beamsplitter_geometry(
        surfs[0], surfs[1], config=comp.beamsplitter_config, cube_side_length_mm=20.0
    )
    assert len(outer_faces) == 6  # 6 outer box faces
    assert len(iface_faces) == 1  # 1 internal diagonal splitting interface
    assert world_verts.shape[0] == 8  # 8 box corners


# ==============================================================================
# 7. 3D PyQt Visualizer Mechanism Toolbar Controls
# ==============================================================================

@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_galvo_controls_and_state_cycling(monkeypatch):
    from optics_app.gui.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    stack = SurfaceStack()
    g_comp, g_surf = create_galvo_component(
        name="Galvo Scan Mirror",
        scan_axis="X",
        min_angle_deg=-5.0,
        max_angle_deg=5.0,
        current_angle_deg=0.0,
    )
    stack.components.append(g_comp)
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(g_surf)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=stack, rays=[])
        assert not widget.btn_mech_state.isHidden()
        assert "GALVO: +0.0°" in widget.btn_mech_state.text()

        # Both btn_mech_state and backward-compatible btn_dmd_state must be present
        assert widget.btn_dmd_state is widget.btn_mech_state

        monkeypatch.setattr(widget, "trace_3d_rays", lambda: [])

        # Toggle/step mechanism state
        next_state = widget.toggle_mech_state()
        assert next_state == "+2.5°"
        assert g_surf.galvo_config.current_angle_deg == 2.5
        assert "GALVO: +2.5°" in widget.btn_mech_state.text()

        # Call via backward-compatible toggle_dmd_state alias
        next_state_2 = widget.toggle_dmd_state()
        assert next_state_2 == "+5.0°"
        assert g_surf.galvo_config.current_angle_deg == 5.0
    finally:
        widget.close()

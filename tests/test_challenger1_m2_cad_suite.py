"""
tests/test_challenger1_m2_cad_suite.py: Authoritative Challenger 1 Adversarial Suite for Milestone 2 CAD Export.

Adversarially tests and verifies:
1. Extreme prism configurations (steep apex angles, unusual aspect ratios, tiny/huge dimensions)
   and Open CASCADE B-Rep solid validity (`solid.is_valid`, volume, closed shells, face count).
2. Cube and plate beam splitters under arbitrary 3D spatial orientations (full SO(3) rotations,
   gimbal lock configurations) and topological manifoldness (closed 2-manifold shells, positive volume).
3. STEP export output files deep inspection:
   - File schema, header/data sections, sequential entity definitions.
   - B-Rep topology entities (MANIFOLD_SOLID_BREP, CLOSED_SHELL, ADVANCED_FACE).
   - Coordinate integrity: every CARTESIAN_POINT is strictly finite (no NaN, Inf).
   - Assembly tree occurrence labels (`Element_X_...`).
   - Roundtrip re-import via Open CASCADE / build123d verifying geometrical validity and volume conservation.
4. Dynamic mechanism kinematics in STEP assembly:
   - Galvo scanner angular sweeps (-20 deg to +20 deg) and pivot offset translations.
   - Mechanism pose placement verification in exported STEP solids.
5. Boundary conditions and selective export filtering.
"""

import os
import re
import math
import tempfile
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
    STEPExporter,
    STEPExportOptions,
    STEPExportEngine,
    create_prism_solid,
    create_beamsplitter_solid,
    get_exportable_elements,
    _get_build123d,
)


@pytest.fixture(scope="module")
def b3d():
    lib = _get_build123d()
    if lib is None:
        pytest.skip("build123d / Open CASCADE CAD kernel is not available")
    return lib


# ==============================================================================
# Domain 1: Extreme Prisms Stress Suite
# ==============================================================================

class TestExtremePrismGeometry:
    """Stress-test prism solid generator across extreme geometries and boundary cases."""

    @pytest.mark.parametrize("apex_deg", [
        0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 14.99,
        15.01, 30.0, 45.0, 60.0, 75.0, 89.9, 90.0, 90.1,
        120.0, 150.0, 175.0, 178.9, 179.5
    ])
    def test_prism_apex_angles_brep_validity(self, b3d, apex_deg):
        """Verify B-Rep validity and positive volume across acute to obtuse apex angles."""
        geom_type = "wedge" if apex_deg <= 15.0 else ("right_angle" if abs(apex_deg - 90.0) < 1.0 else "equilateral")
        cfg = PrismConfig(
            geometry_type=geom_type,
            apex_angle_deg=apex_deg,
            base_width_mm=25.0,
            length_mm=25.0,
            height_mm=25.0,
        )
        s1, s2 = Surface(), Surface()
        solid = create_prism_solid([s1, s2], config=cfg, apex_angle_deg=apex_deg, geometry_type=geom_type)

        assert solid is not None, f"create_prism_solid returned None for apex={apex_deg}"
        assert solid.is_valid, f"Prism B-Rep solid invalid for apex={apex_deg}"
        assert solid.volume > 0, f"Prism volume <= 0 ({solid.volume}) for apex={apex_deg}"
        assert len(solid.shells()) == 1, f"Prism must have exactly 1 closed shell, got {len(solid.shells())}"
        assert len(solid.faces()) >= 4, f"Prism must have >= 4 faces, got {len(solid.faces())}"

    @pytest.mark.parametrize("bw, w, h", [
        (0.05, 500.0, 50.0),      # Razor-thin base, ultra-long extrusion (aspect 1:10000)
        (500.0, 0.05, 50.0),      # Wide base, ultra-thin extrusion
        (1.0, 10.0, 500.0),       # Tall tower prism
        (500.0, 500.0, 0.1),      # Ultra-flat pancake prism
        (0.01, 0.01, 0.01),       # Microscopic millimeter-scale prism
        (1000.0, 1000.0, 1000.0), # Macroscopic meter-scale prism
    ])
    def test_prism_extreme_aspect_ratios(self, b3d, bw, w, h):
        """Verify B-Rep manifold validity under extreme aspect ratio permutations."""
        cfg = PrismConfig(
            geometry_type="Equilateral",
            apex_angle_deg=60.0,
            base_width_mm=bw,
            length_mm=w,
            height_mm=h,
        )
        s1, s2 = Surface(), Surface()
        solid = create_prism_solid([s1, s2], config=cfg, base_width_mm=bw, length_mm=w, height_mm=h)

        assert solid is not None
        assert solid.is_valid
        assert solid.volume > 0
        assert len(solid.shells()) == 1
        assert len(solid.faces()) >= 4

    def test_prism_degenerate_inputs_fallback(self, b3d):
        """Verify robust fallback for degenerate or unphysical parameters (negative or zero)."""
        degenerate_cases = [
            {"base_width_mm": 0.0},
            {"base_width_mm": -10.0},
            {"length_mm": 0.0},
            {"height_mm": 0.0},
            {"apex_angle_deg": 0.0},
            {"apex_angle_deg": -45.0},
        ]
        s1, s2 = Surface(), Surface()
        for kwargs in degenerate_cases:
            solid = create_prism_solid([s1, s2], **kwargs)
            assert solid is not None, f"Fallback failed for {kwargs}"
            assert solid.is_valid, f"Fallback solid is not valid for {kwargs}"
            assert solid.volume > 0, f"Fallback volume <= 0 for {kwargs}"


# ==============================================================================
# Domain 2: Beam Splitters Orientation & Topological Manifoldness Suite
# ==============================================================================

class TestBeamSplitterTopologicalManifoldness:
    """Stress-test Cube and Plate beam splitters under spatial orientations and manifold checks."""

    SPATIAL_ORIENTATIONS = [
        (0.0, 0.0, 0.0),          # Identity
        (45.0, 0.0, 0.0),         # Pure roll
        (0.0, 45.0, 0.0),         # Pure pitch
        (0.0, 0.0, 45.0),         # Pure yaw
        (35.264, 45.0, 30.0),     # Compound 3D tilt
        (90.0, 0.0, 0.0),         # 90 deg rotation
        (0.0, 90.0, 0.0),         # Gimbal singularity pitch
        (180.0, 0.0, 0.0),        # Inversion around X
        (0.0, 180.0, 0.0),        # Inversion around Y
        (0.0, 0.0, 180.0),        # Inversion around Z
        (-135.0, -45.0, 120.0),   # Negative compound angles
        (-179.9, 89.9, -45.0),    # Boundary Euler angles
    ]

    @pytest.mark.parametrize("angles", SPATIAL_ORIENTATIONS)
    def test_cube_beamsplitter_manifoldness_under_orientations(self, b3d, angles):
        """Verify Cube beam splitter compound solid preserves manifoldness and volume under SO(3) rotations."""
        L = 25.4
        cfg = BeamSplitterConfig(splitter_type="Cube", cube_side_length_mm=L)
        s1 = Surface()
        s1.name = "Cube Beam Splitter"
        cube = create_beamsplitter_solid([s1], config=cfg)

        assert cube is not None
        assert cube.is_valid
        assert len(cube.children) == 2, "Cube beam splitter must consist of 2 prism halves"

        # Apply 3D spatial transformation
        loc = b3d.Location((15.0, -25.0, 100.0), angles)
        oriented_cube = cube.locate(loc)

        assert oriented_cube.is_valid, f"Oriented cube invalid at {angles}"
        assert oriented_cube.volume == pytest.approx(L**3, rel=1e-3)

        # Inspect both constituent prism halves
        h1 = oriented_cube.children[0]
        h2 = oriented_cube.children[1]

        assert h1.is_valid and h2.is_valid
        assert len(h1.shells()) == 1 and len(h2.shells()) == 1, "Prism halves must be 2-manifold closed shells"
        assert h1.volume == pytest.approx(0.5 * L**3, rel=1e-3)
        assert h2.volume == pytest.approx(0.5 * L**3, rel=1e-3)
        assert len(h1.faces()) == 5 and len(h2.faces()) == 5, "Right-angle prism half must have 5 planar faces"

    @pytest.mark.parametrize("angles", SPATIAL_ORIENTATIONS)
    def test_plate_beamsplitter_manifoldness_under_orientations(self, b3d, angles):
        """Verify Plate beam splitter (circular and rectangular) preserves manifoldness under SO(3) rotations."""
        # 1. Circular plate
        cfg_circ = BeamSplitterConfig(splitter_type="Plate", substrate_thickness_mm=2.5, clear_aperture_mm=25.4)
        s_circ = Surface()
        s_circ.semi_diameter = 12.7
        circ_solid = create_beamsplitter_solid([s_circ], config=cfg_circ)

        loc = b3d.Location((-10.0, 20.0, 50.0), angles)
        oriented_circ = circ_solid.locate(loc)
        assert oriented_circ.is_valid
        assert len(oriented_circ.shells()) == 1
        expected_circ_vol = math.pi * (12.7**2) * 2.5
        assert oriented_circ.volume == pytest.approx(expected_circ_vol, rel=1e-3)

        # 2. Rectangular plate
        cfg_rect = BeamSplitterConfig(splitter_type="Plate", substrate_thickness_mm=3.0)
        s_rect = Surface()
        s_rect.aperture_type = "Rectangular"
        s_rect.aperture_x_half = 15.0
        s_rect.aperture_y_half = 10.0
        rect_solid = create_beamsplitter_solid([s_rect], config=cfg_rect, clear_aperture_x=30.0, clear_aperture_y=20.0)

        oriented_rect = rect_solid.locate(loc)
        assert oriented_rect.is_valid
        assert len(oriented_rect.shells()) == 1
        expected_rect_vol = 30.0 * 20.0 * 3.0
        assert oriented_rect.volume == pytest.approx(expected_rect_vol, rel=1e-3)

    def test_plate_extreme_aspect_ratios(self, b3d):
        """Verify plate beam splitter with extreme thickness-to-aperture ratios."""
        s = Surface()
        s.semi_diameter = 50.0
        # Ultra-thin pellicle-like substrate
        thin_solid = create_beamsplitter_solid([s], substrate_thickness_mm=0.1, thickness=0.1)
        assert thin_solid.is_valid
        assert thin_solid.volume > 0

        # Thick transmission block
        thick_solid = create_beamsplitter_solid([s], substrate_thickness_mm=80.0, thickness=80.0)
        assert thick_solid.is_valid
        assert thick_solid.volume > 0


# ==============================================================================
# Domain 3: STEP Export Verification & Deep File Inspection Suite
# ==============================================================================

class TestStepExportDeepInspection:
    """Adversarially verify STEP output files: syntax, entities, coordinates, assembly tree, roundtrip."""

    def _build_complex_heterogeneous_system(self):
        """Build optical system containing singlets, prisms, cube BS, plate BS, galvo, and image plane."""
        stack = SurfaceStack()

        # Surface 0: Object
        s0 = Surface()
        s0.name = "Object Plane"
        s0.thickness = 30.0
        stack.add_surface(s0)

        # Component 1: Lens Singlet
        s1 = Surface()
        s1.name = "Lens Front"
        s1.radius = 50.0
        s1.thickness = 5.0
        s1.glass = "N-BK7"
        s1.semi_diameter = 12.7

        s2 = Surface()
        s2.name = "Lens Rear"
        s2.radius = -50.0
        s2.thickness = 25.0
        s2.semi_diameter = 12.7
        stack.add_surface(s1)
        stack.add_surface(s2)

        # Component 2: Equilateral Prism
        p_comp, p_surfs = create_prism_component(
            name="Equilateral_Prism",
            geometry_type="Equilateral",
            apex_angle_deg=60.0,
            base_width_mm=25.0,
            glass="N-SF11",
        )
        for s in p_surfs:
            stack.add_surface(s)
        stack.components.append(p_comp)

        # Component 3: Cube Beam Splitter
        bs_comp, bs_surfs = create_beamsplitter_component(
            name="Cube_Splitter",
            splitter_type="Cube",
        )
        for s in bs_surfs:
            stack.add_surface(s)
        stack.components.append(bs_comp)

        # Component 4: Galvo Scanner
        galvo_comp, galvo_surf = create_galvo_component(
            name="Galvo_Scanner_X",
            scan_axis="X",
            current_angle_deg=10.0,
            pivot_offset_mm=[0.0, 5.0, -2.0],
        )
        stack.add_surface(galvo_surf)
        stack.components.append(galvo_comp)

        # Surface final: Image plane
        s_img = Surface()
        s_img.name = "Detector"
        s_img.surface_type = SurfaceType.DETECTOR
        s_img.semi_diameter = 15.0
        stack.add_surface(s_img)

        return stack

    def test_step_export_structure_and_entity_definitions(self, b3d):
        """Deeply inspect generated STEP file syntax and standard entity definitions."""
        stack = self._build_complex_heterogeneous_system()

        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, "stress_optical_assembly.step")
            res = STEPExporter.export(stack, step_file)

            assert res.is_success
            assert res.num_components_exported >= 4
            assert os.path.exists(step_file)
            assert os.path.getsize(step_file) > 5000  # Non-trivial CAD file

            with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 1. Standard framing headers
            assert content.startswith("ISO-10303-21;"), "Must begin with ISO-10303-21 header"
            assert "HEADER;" in content and "ENDSEC;" in content, "Must contain valid HEADER block"
            assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));" in content or "FILE_SCHEMA" in content
            assert "DATA;" in content
            assert content.strip().endswith("END-ISO-10303-21;"), "Must terminate with END-ISO-10303-21;"

            # 2. B-Rep Topological Entities
            solids_count = len(re.findall(r"MANIFOLD_SOLID_BREP", content))
            shells_count = len(re.findall(r"CLOSED_SHELL", content))
            faces_count = len(re.findall(r"ADVANCED_FACE", content))
            placements_count = len(re.findall(r"AXIS2_PLACEMENT_3D", content))

            assert solids_count >= res.num_components_exported, f"Expected >= {res.num_components_exported} solids, got {solids_count}"
            assert shells_count >= solids_count, "Every solid must have at least one closed shell"
            assert faces_count >= 20, f"Expected complex polyhedral faces count >= 20, got {faces_count}"
            assert placements_count >= res.num_components_exported

            # 3. Assembly labels and occurrence prefixes
            assert re.search(r"Element_\d+_Lens", content)
            assert re.search(r"Element_\d+_Equilateral_Prism", content)
            assert re.search(r"Element_\d+_Cube_Splitter", content)

    def test_step_cartesian_coordinates_strictly_finite(self, b3d):
        """Verify that every CARTESIAN_POINT in the exported STEP file contains finite, valid coordinates."""
        stack = self._build_complex_heterogeneous_system()

        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, "coords_check.step")
            res = STEPExporter.export(stack, step_file)
            assert res.is_success

            with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Find all CARTESIAN_POINT definitions: CARTESIAN_POINT ( '...', ( x, y, z ) )
            points = re.findall(r"CARTESIAN_POINT\s*\(\s*'.*?'\s*,\s*\(([^)]+)\)\s*\)", content)
            assert len(points) > 50, "Expected numerous CARTESIAN_POINT entities"

            coords_all = []
            for pt in points:
                coords = [float(v.strip()) for v in pt.split(",")]
                assert len(coords) in (2, 3), f"Cartesian point must be 2D or 3D, got {coords}"
                for c in coords:
                    assert not math.isnan(c), f"Found NaN coordinate in {pt}"
                    assert not math.isinf(c), f"Found Inf coordinate in {pt}"
                    assert abs(c) < 1e6, f"Unreasonably large coordinate: {c}"
                coords_all.append(coords)

            coords_3d = [c for c in coords_all if len(c) == 3]
            coords_np = np.array(coords_3d)
            bbox_span = coords_np.max(axis=0) - coords_np.min(axis=0)
            assert bbox_span[0] > 1.0, "Assembly must have non-zero X dimension"
            assert bbox_span[1] > 1.0, "Assembly must have non-zero Y dimension"
            assert bbox_span[2] > 10.0, "Assembly must have non-zero longitudinal Z span"

    def test_step_cad_kernel_roundtrip_reimport(self, b3d):
        """Verify roundtrip: export STEP assembly and re-import via Open CASCADE CAD kernel."""
        stack = self._build_complex_heterogeneous_system()

        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, "roundtrip.step")
            res = STEPExporter.export(stack, step_file)
            assert res.is_success

            # Re-import back using Open CASCADE / build123d
            imported = b3d.import_step(step_file)
            assert imported is not None, "Failed to re-import exported STEP file into build123d"
            assert imported.is_valid, "Imported Open CASCADE assembly is geometrically invalid"
            assert imported.volume > 0, f"Imported assembly volume must be positive, got {imported.volume}"
            assert len(imported.solids()) >= 4, f"Expected at least 4 solids upon re-import, got {len(imported.solids())}"


# ==============================================================================
# Domain 4: Kinematic Mechanisms in STEP Assembly
# ==============================================================================

class TestMechanismDynamicKinematicsInStep:
    """Stress-test dynamic actuator poses (scan angles, pivot offsets) in STEP export."""

    @pytest.mark.parametrize("scan_angle", [-20.0, -10.0, 0.0, 10.0, 20.0])
    def test_galvo_scan_angle_variation_shifts_step_geometry(self, b3d, scan_angle):
        """Verify that varying galvo scan angle strictly changes mirror pose in exported STEP."""
        stack = SurfaceStack()
        s0 = Surface()
        s0.thickness = 50.0
        stack.add_surface(s0)

        galvo_comp, galvo_surf = create_galvo_component(
            name="Galvo_Scan",
            scan_axis="X",
            current_angle_deg=scan_angle,
            pivot_offset_mm=[0.0, 0.0, -2.0],
        )
        stack.add_surface(galvo_surf)
        stack.components.append(galvo_comp)

        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, f"galvo_{int(scan_angle)}.step")
            res = STEPExporter.export(stack, step_file)
            assert res.is_success

            # Re-import and check mirror orientation
            imported = b3d.import_step(step_file)
            assert imported.is_valid
            solids = imported.solids()
            assert len(solids) == 1
            mirror_solid = solids[0]

            # The center of mass / bounding box must vary predictably with angle
            center = mirror_solid.center()
            # For scan_axis="X", mirror rotates around Y, tilting in X-Z plane
            if abs(scan_angle) > 1e-3:
                assert abs(center.X) > 0.01, f"Expected non-zero X shift at {scan_angle} deg, got {center.X}"
            else:
                assert abs(center.X) < 1e-4, f"Expected near-zero X shift at 0 deg, got {center.X}"

    def test_galvo_angular_sweep_centroid_divergence(self, b3d):
        """Verify that solids at +20 deg and -20 deg are distinct in 3D space."""
        def get_solid_center_for_angle(deg):
            stack = SurfaceStack()
            s0 = Surface()
            s0.thickness = 30.0
            stack.add_surface(s0)
            comp, surf = create_galvo_component(
                name="G",
                scan_axis="X",
                current_angle_deg=deg,
                pivot_offset_mm=[0.0, 0.0, -5.0],
            )
            stack.add_surface(surf)
            stack.components.append(comp)

            with tempfile.TemporaryDirectory() as tmpdir:
                sf = os.path.join(tmpdir, "g.step")
                STEPExporter.export(stack, sf)
                imp = b3d.import_step(sf)
                return imp.solids()[0].center()

        c_pos = get_solid_center_for_angle(20.0)
        c_neg = get_solid_center_for_angle(-20.0)

        # Centers must not be identical due to pivot lever arm rotation
        dist_pos_neg = math.sqrt((c_pos.X - c_neg.X)**2 + (c_pos.Y - c_neg.Y)**2 + (c_pos.Z - c_neg.Z)**2)
        assert dist_pos_neg > 0.5, f"Galvo scan rotation must physically shift mirror center, got dist={dist_pos_neg}"


# ==============================================================================
# Domain 5: Boundary & Selective Export Resiliency
# ==============================================================================

class TestExportResiliency:
    """Verify edge-case handling, empty stack, and partial element selection."""

    def test_empty_surface_stack_export(self, b3d):
        """Exporting an empty stack produces a valid degenerate STEP header without crashing."""
        stack = SurfaceStack()
        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, "empty.step")
            res = STEPExporter.export(stack, step_file)
            assert res.is_success
            assert res.num_components_exported == 0
            assert os.path.exists(step_file)
            with open(step_file, "r") as f:
                content = f.read()
            assert "ISO-10303-21;" in content
            assert "END-ISO-10303-21;" in content

    def test_selective_element_export(self, b3d):
        """Exporting only selected element indices exports strictly those components."""
        stack = SurfaceStack()
        s0 = Surface()
        s0.thickness = 20.0
        stack.add_surface(s0)

        p_comp, p_surfs = create_prism_component(name="Prism", base_width_mm=20.0)
        for s in p_surfs:
            stack.add_surface(s)
        stack.components.append(p_comp)

        bs_comp, bs_surfs = create_beamsplitter_component(name="CubeBS", splitter_type="Cube")
        for s in bs_surfs:
            stack.add_surface(s)
        stack.components.append(bs_comp)

        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = os.path.join(tmpdir, "selected.step")
            # Select ONLY element index 0 (the prism, since s0 is object)
            res = STEPExporter.export(stack, step_file, selected_element_indices=[0])
            assert res.is_success
            assert res.num_components_exported == 1

            imp = b3d.import_step(step_file)
            assert imp.is_valid
            assert len(imp.solids()) == 1

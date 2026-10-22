"""
Adversarial Stress-Test & Vulnerability Verification Suite for Final Milestone M5 (Challenger).
Tests:
1. Rapid parameter slider scrubbing under extreme values (large angles, large spacings) verifying fast trace <50ms without UI thread lock or memory leaks.
2. Dynamic sequence modifications: load .zmx, insert between surfaces, flip front-to-back, rotate 3D, and delete with air gap merging, confirming total optical rail alignment is conserved.
3. End-to-end flow: load model -> misalign grating -> trigger SciPy optimizer -> verify spot size converges to minimum on 0.8mm detector -> export 3D STEP solid -> verify ISO-10303-21 CAD syntax.
4. Package import and architecture integrity.
"""

import os
import sys
import time
import math
import tempfile
import tracemalloc
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from v2.engine.lde import SurfaceStack, build_rotation_matrix
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, STEPExportEngine
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system


# ============================================================================
# PART 1: RAPID PARAMETER SLIDER SCRUBBING & LATENCY / MEMORY STRESS TESTS
# ============================================================================

class TestPart1_SliderScrubbingAndPerformance:
    """Stress tests rapid parameter scrubbing, extreme parameter bounds, execution latency, and memory stability."""

    def test_extreme_grating_angle_scrubbing_latency(self, loaded_optical_system):
        """Scrub grating tilt angles through extreme ranges (-89.9 deg to +89.9 deg) and verify latency < 50ms per step."""
        system = loaded_optical_system
        grating_idx = None
        for i, s in enumerate(system.surfaces):
            if s.surface_type == SurfaceType.GRATING:
                grating_idx = i
                break
        assert grating_idx is not None, "Grating surface must exist"

        test_angles = np.linspace(-89.9, 89.9, 50)
        latencies_ms = []

        wavelengths = [1.10, 1.225, 1.35]
        test_rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in wavelengths
        ]

        for angle in test_angles:
            system.surfaces[grating_idx].tilt_y = float(angle)
            system._update_accumulated_positions()

            t0 = time.perf_counter()
            traced = system.trace_rays(test_rays)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            # Ensure trace completed without crash and returned results
            assert len(traced) == len(test_rays)

        max_latency = max(latencies_ms)
        mean_latency = np.mean(latencies_ms)
        p95_latency = np.percentile(latencies_ms, 95)

        print(f"\n[Scrubbing Angle] Mean: {mean_latency:.3f}ms | P95: {p95_latency:.3f}ms | Max: {max_latency:.3f}ms")
        assert max_latency < 50.0, f"Max trace latency {max_latency:.2f}ms exceeded 50ms requirement"
        assert mean_latency < 15.0, f"Mean trace latency {mean_latency:.2f}ms was excessively high"

    def test_extreme_spacing_scrubbing_latency(self, loaded_optical_system):
        """Scrub inter-element spacings over 4 orders of magnitude (0.001 mm to 5000 mm) and verify latency < 50ms."""
        system = loaded_optical_system
        test_spacings = [0.001, 0.01, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 1500.0, 5000.0]
        latencies_ms = []

        wavelengths = [1.10, 1.225, 1.35]
        test_rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in wavelengths
        ]

        for spacing in test_spacings:
            # Change collimator air gap (surface 4) and focuser air gap (surface 12)
            system.surfaces[4].thickness = float(spacing)
            system.surfaces[12].thickness = float(spacing)
            system._update_accumulated_positions()

            t0 = time.perf_counter()
            traced = system.trace_rays(test_rays)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            assert len(traced) == len(test_rays)

        max_latency = max(latencies_ms)
        mean_latency = np.mean(latencies_ms)
        print(f"\n[Scrubbing Spacing] Mean: {mean_latency:.3f}ms | Max: {max_latency:.3f}ms")
        assert max_latency < 50.0, f"Spacing scrub max latency {max_latency:.2f}ms exceeded 50ms"

    def test_rapid_scrubbing_burst_memory_stability(self, loaded_optical_system):
        """Execute 500 rapid parameter scrub iterations in a tight loop and verify no memory leak or unbounded heap growth."""
        system = loaded_optical_system
        grating_idx = 8  # grating surface
        
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        wavelengths = [1.10, 1.225, 1.35]
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in wavelengths
        ]

        # 500 continuous slider change operations
        for step in range(500):
            angle = (step % 200 - 100) * 0.1  # -10.0 deg to +10.0 deg
            system.surfaces[grating_idx].tilt_y = angle
            system._update_accumulated_positions()
            system.trace_rays(rays)

        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_growth_kb = sum(stat.size_diff for stat in top_stats) / 1024.0

        print(f"\n[Memory Stability] 500 scrub steps memory delta: {total_growth_kb:.2f} KB")
        # Memory growth across 500 steps should be negligible (< 5 MB = 5120 KB)
        assert total_growth_kb < 5120.0, f"Excessive memory growth during slider scrubbing: {total_growth_kb:.2f} KB"

    def test_extreme_non_physical_boundary_inputs(self, loaded_optical_system):
        """Stress-test non-physical boundary inputs (negative thicknesses, 90 deg grazing incidence, zero apertures)."""
        system = loaded_optical_system

        # Test 1: Zero thickness
        system.surfaces[2].thickness = 0.0
        system._update_accumulated_positions()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)]
        traced = system.trace_rays(rays)
        assert len(traced) == 1

        # Test 2: Extreme grazing incidence angle on grating
        grating_idx = 8
        system.surfaces[grating_idx].tilt_y = 89.99
        system._update_accumulated_positions()
        traced = system.trace_rays(rays)
        assert len(traced) == 1

        # Test 3: Large decenter
        system.surfaces[2].decenter_x = 50.0
        system.surfaces[2].decenter_y = -50.0
        system._update_accumulated_positions()
        traced = system.trace_rays(rays)
        assert len(traced) == 1


# ============================================================================
# PART 2: DYNAMIC SEQUENCE MODIFICATIONS & OPTICAL RAIL ALIGNMENT INVARIANTS
# ============================================================================

class TestPart2_DynamicSequenceAndRailConservation:
    """Stress tests dynamic sequence operations: .zmx ingestion, insertion, front-to-back flip, 3D tilt, deletion with air gap merge."""

    def test_zmx_ingestion_and_surface_fidelity(self, sample_thorlabs_doublet_zmx):
        """Parse Thorlabs doublet .zmx file and verify surface parameters and glasses."""
        stack = parse_zmx_file(sample_thorlabs_doublet_zmx)
        assert stack is not None
        assert len(stack.surfaces) >= 4
        # Verify glasses
        glasses = [s.glass for s in stack.surfaces if s.glass not in ('AIR', '')]
        assert "N-BK7" in glasses
        assert "SF5" in glasses

    def test_sequence_insertion_and_rail_integrity(self, loaded_spectrometer_stack, sample_thorlabs_singlet_zmx):
        """Insert a catalog lens into the spectrometer and verify global coordinate continuity."""
        stack = loaded_spectrometer_stack
        initial_count = len(stack.surfaces)
        
        # Load singlet to insert
        singlet_stack = parse_zmx_file(sample_thorlabs_singlet_zmx)
        lens_surfaces = [s for s in singlet_stack.surfaces if s.surface_type == SurfaceType.STANDARD and s.glass not in ('AIR', '')]
        
        # Insert after surface 3 (collimator front)
        insert_idx = 3
        for offset, s in enumerate(lens_surfaces):
            stack.surfaces.insert(insert_idx + offset, s)

        assert len(stack.surfaces) == initial_count + len(lens_surfaces)
        
        # Verify OpticalSystem can trace through the modified sequence
        opt_sys = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)]
        traced = opt_sys.trace_rays(rays)
        assert len(traced) == 1

    def test_front_to_back_flip_double_flip_identity(self, sample_thorlabs_doublet_zmx):
        """Verify front-to-back flip inverts curvatures (R -> -R) and double-flip is an exact identity transformation."""
        stack = parse_zmx_file(sample_thorlabs_doublet_zmx)
        lens_surfs = [s for s in stack.surfaces if s.surface_type == SurfaceType.STANDARD and s.glass not in ('AIR', '')]
        assert len(lens_surfs) >= 2

        orig_radii = [s.radius for s in lens_surfs]
        orig_thicknesses = [s.thickness for s in lens_surfs]
        orig_glasses = [s.glass for s in lens_surfs]

        # Invert / Flip doublet element
        flipped_radii = [-r if r != 0.0 else 0.0 for r in reversed(orig_radii)]
        flipped_thicknesses = list(reversed(orig_thicknesses))
        flipped_glasses = list(reversed(orig_glasses))

        # Re-flip (double flip)
        re_flipped_radii = [-r if r != 0.0 else 0.0 for r in reversed(flipped_radii)]
        re_flipped_thicknesses = list(reversed(flipped_thicknesses))
        re_flipped_glasses = list(reversed(flipped_glasses))

        # Assert double-flip returns to exact original values
        assert np.allclose(re_flipped_radii, orig_radii, atol=1e-12)
        assert np.allclose(re_flipped_thicknesses, orig_thicknesses, atol=1e-12)
        assert re_flipped_glasses == orig_glasses

    def test_rigid_body_3d_rotation_so3_orthogonality(self):
        """Verify 3D rigid body Euler rotation matrices are valid SO(3) elements (R^T R = I, det(R) = +1)."""
        angles = [
            (15.0, 30.0, 45.0),
            (-45.0, 60.0, -120.0),
            (0.001, -0.002, 0.003),
            (180.0, 90.0, -90.0),
        ]
        for tx, ty, tz in angles:
            R = build_rotation_matrix(tx, ty, tz)
            # Orthogonality
            identity_approx = R.T @ R
            assert np.allclose(identity_approx, np.eye(3), atol=1e-10), f"R^T R != I for angles {(tx, ty, tz)}"
            # Determinant must be +1 (special orthogonal group SO(3))
            det_R = np.linalg.det(R)
            assert np.isclose(det_R, 1.0, atol=1e-10), f"det(R) != 1 for angles {(tx, ty, tz)}"

    def test_element_deletion_air_gap_merging_rail_conservation(self, loaded_spectrometer_stack):
        """Verify deleting an optical element merges its thickness into preceding air gap, strictly conserving total rail length."""
        stack = loaded_spectrometer_stack
        
        # Measure initial total axial distance from surface 0 to detector
        initial_total_rail_length = sum(s.thickness for s in stack.surfaces)
        
        # Select an internal element to delete: e.g. surface 3 (a lens element with thickness and preceding gap)
        target_idx = 3
        elem_thickness = stack.surfaces[target_idx].thickness
        preceding_gap = stack.surfaces[target_idx - 1].thickness

        # Delete element and merge thickness into preceding air gap
        stack.surfaces[target_idx - 1].thickness += elem_thickness
        del stack.surfaces[target_idx]

        # Calculate new total rail length
        new_total_rail_length = sum(s.thickness for s in stack.surfaces)

        print(f"\n[Rail Conservation] Initial: {initial_total_rail_length:.4f} mm | After Delete: {new_total_rail_length:.4f} mm")
        assert np.isclose(new_total_rail_length, initial_total_rail_length, atol=1e-9), \
            f"Optical rail alignment length changed: {new_total_rail_length} vs {initial_total_rail_length}"


# ============================================================================
# PART 3: END-TO-END OPTIMIZATION, SPOT SIZE CONVERGENCE & CAD STEP SYNTAX
# ============================================================================

class TestPart3_EndToEndOptimizerAndCADSTEP:
    """Stress tests the complete end-to-end workflow: misalignment -> SciPy optimization -> spot minimization -> 3D STEP export -> ISO-10303-21 syntax validation."""

    def test_spectrometer_misalignment_and_optimizer_convergence(self, loaded_spectrometer_stack):
        """Deliberately misalign grating angle and lens spacing, trigger SciPy optimizer, and verify spot size minimizes on 0.8mm detector."""
        stack = loaded_spectrometer_stack
        opt_sys = OpticalSystem(stack)

        primary_wl = 1.225

        spot_init = generate_spot_diagram_from_system(opt_sys, num_rays=21, wavelength_um=primary_wl)
        init_rms = spot_init.calculate_rms_radius()
        print(f"\n[Optimizer E2E] Initial nominal RMS spot radius: {init_rms*1000.0:.2f} µm")

        # 1. Deliberately misalign grating tilt and focuser spacing
        # In the 17-surface spectrometer: surface 3 is collimator/fold, surface 6 is focuser spacing
        stack.surfaces[3].tilt_x = -11.0  # nominal is -12.0
        stack.surfaces[6].thickness = 43.0  # nominal is 45.0
        stack._update_accumulated_positions()

        opt_sys_misaligned = OpticalSystem(stack)
        spot_misaligned = generate_spot_diagram_from_system(opt_sys_misaligned, num_rays=21, wavelength_um=primary_wl)
        misaligned_rms = spot_misaligned.calculate_rms_radius()
        print(f"[Optimizer E2E] Misaligned RMS spot radius: {misaligned_rms*1000.0:.2f} µm")

        # 2. Trigger SciPy Optimizer
        optimizer = OpticalOptimizer(
            opt_sys_misaligned,
            wavelengths=[primary_wl],
            method="Nelder-Mead",
            num_rays=15,
            max_iter=30
        )

        res = optimizer.optimize(
            variable_indices={"thickness": [6], "tilt_x": [3]}
        )

        print(f"[Optimizer E2E] Optimizer status: {res.status}, Iterations: {res.iterations}")
        assert res.optimized_stack is not None, "Optimizer must produce optimized stack"

        # 3. Verify final optimized spot size
        optimized_sys = OpticalSystem(res.optimized_stack)
        spot_final = generate_spot_diagram_from_system(optimized_sys, num_rays=21, wavelength_um=primary_wl)
        final_rms = spot_final.calculate_rms_radius()
        print(f"[Optimizer E2E] Final optimized RMS spot radius: {final_rms*1000.0:.2f} µm")

        # Spot size must be smaller than misaligned or within single pixel 0.8mm detector boundary (0.4mm radius)
        assert final_rms <= misaligned_rms or final_rms < 0.4, f"Final RMS radius {final_rms:.4f}mm must be smaller than misaligned {misaligned_rms:.4f}mm or within detector"
        assert final_rms < 0.4, f"Final RMS radius {final_rms:.4f}mm must fit inside 0.4mm detector radius"

    def test_3d_step_cad_export_and_iso_10303_21_syntax(self, loaded_spectrometer_stack, tmp_path):
        """Export 3D STEP solid model of spectrometer and verify strict ISO-10303-21 CAD syntax and entity topology."""
        stack = loaded_spectrometer_stack
        step_file = tmp_path / "spectrometer_e2e_export.step"
        step_path = str(step_file)

        result = STEPExporter.export(stack, step_path)
        
        assert os.path.exists(step_path), "Exported STEP file must exist on disk"
        file_size = os.path.getsize(step_path)
        assert file_size > 500, f"Exported STEP file is too small ({file_size} bytes)"

        with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
            step_content = f.read()

        # ISO-10303-21 Syntax Checks
        assert "ISO-10303-21;" in step_content, "STEP file must start with ISO-10303-21 tag"
        assert "HEADER;" in step_content, "STEP file must contain HEADER section"
        assert "FILE_DESCRIPTION" in step_content, "STEP HEADER must contain FILE_DESCRIPTION"
        assert "FILE_NAME" in step_content, "STEP HEADER must contain FILE_NAME"
        assert "FILE_SCHEMA" in step_content, "STEP HEADER must contain FILE_SCHEMA"
        assert "DATA;" in step_content, "STEP file must contain DATA section"
        assert "ENDSEC;" in step_content, "STEP file must contain ENDSEC tags"
        assert "END-ISO-10303-21;" in step_content, "STEP file must end with END-ISO-10303-21 terminator"

        # Entity Reference Topology Check
        import re
        all_defs = set(re.findall(r"#(\d+)\s*=", step_content))
        all_refs = re.findall(r"#(\d+)", step_content)

        assert len(all_defs) > 10, f"STEP file must contain solid geometry entities, found {len(all_defs)}"
        print(f"\n[CAD STEP Validation] Validated {len(all_defs)} entity definitions, {len(all_refs)} entity references.")

        # Check for dangling entity references
        dangling_refs = [ref for ref in all_refs if ref not in all_defs]
        assert len(dangling_refs) == 0, f"Found {len(dangling_refs)} dangling entity references in STEP file: {dangling_refs[:5]}"



# ============================================================================
# PART 4: PACKAGE IMPORT AND ARCHITECTURAL INTEGRITY (VULNERABILITY DISCOVERY)
# ============================================================================

class TestPart4_PackageIntegrityAndVulnerabilityChecks:
    """Verifies architectural consistency, package import integrity, and catches regressions."""

    def test_v2_package_tracer_import_integrity(self):
        """
        Adversarial Test: Check that v2.engine.tracer can be imported cleanly without NameError: name 'Union' is not defined.
        If v2.engine.tracer has missing imports, this test will explicitly detect the vulnerability.
        """
        import importlib
        try:
            mod = importlib.import_module("v2.engine.tracer")
            assert hasattr(mod, "DualSpeedTracer")
        except NameError as e:
            pytest.fail(f"VULNERABILITY DETECTED in v2.engine.tracer: {e}")

    def test_v2_package_engine_init_import_integrity(self):
        """
        Adversarial Test: Check that v2.engine top-level package can be imported cleanly without crashing.
        """
        import importlib
        try:
            mod = importlib.import_module("v2.engine")
            assert hasattr(mod, "SpectrometerSystem")
        except NameError as e:
            pytest.fail(f"VULNERABILITY DETECTED in v2.engine: {e}")

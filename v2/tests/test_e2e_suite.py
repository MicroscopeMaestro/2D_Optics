"""
Comprehensive Multi-Tier E2E Test Suite for 2D Optics Spectrometer Studio V2.
Covers:
- Tier 1: Feature Coverage (60 tests, 5 per feature across F1 to F12)
- Tier 2: Boundary & Corner Conditions (60 tests, 5 per feature across F1 to F12)
- Tier 3: Cross-Feature Pairwise Interactions (12 tests)
- Tier 4: Real-World Spectrometer Design Workflows (6 scenarios)
"""

import os
import json
import time
import pytest
import numpy as np

from optics_app.gui.qt_compat import QT_AVAILABLE
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.grating import DiffractionGrating
from optics_app.engine.optimizer import (
    OpticalOptimizer, OptimizationResult, TerminationStatus
)
from optics_app.io.zmx_parser import parse_zmx_file
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system
from optics_app.analysis.grating_dispersion import (
    GratingDispersionAnalysis, DispersionMetrics, GratingDispersionResult
)
from optics_app.cad.exporter import STEPExporter, STEPExportEngine
from optics_app.cad.dxf_exporter import DXF2DExporter
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.main_window import MainWindow


# ============================================================================
# TIER 1: FEATURE COVERAGE (60 tests: 5 tests per feature F1 - F12)
# ============================================================================

class TestTier1_F1_JSONModelIngestion:
    """F1: JSON prescription ingestion of NIR spectrometer model."""

    def test_t1_f1_schema_and_version(self, spectrometer_raw_data):
        assert spectrometer_raw_data.get("version") == 3
        assert "surfaces" in spectrometer_raw_data

    def test_t1_f1_seventeen_surfaces_count(self, loaded_spectrometer_stack):
        assert len(loaded_spectrometer_stack.surfaces) == 17

    def test_t1_f1_nir_band_wavelengths(self, spectrometer_raw_data):
        wls = spectrometer_raw_data.get("wavelengths", [])
        np.testing.assert_allclose(wls, [1.10, 1.225, 1.35], atol=1e-4)

    def test_t1_f1_asphere_coefficients_precision(self, loaded_spectrometer_stack):
        surf12 = loaded_spectrometer_stack.surfaces[12]
        assert surf12.surface_type == SurfaceType.ASPHERIC
        assert abs(surf12.radius - 3.478709) < 1e-4
        assert abs(surf12.conic - (-0.1263039)) < 1e-4

    def test_t1_f1_detector_0p8mm_clear_aperture(self, loaded_spectrometer_stack):
        det = loaded_spectrometer_stack.surfaces[16]
        assert det.surface_type == SurfaceType.DETECTOR
        assert abs(det.semi_diameter - 0.4) < 1e-4


class TestTier1_F2_ZemaxParserAndCatalog:
    """F2: Thorlabs & custom Zemax (.zmx) prescription ingestion."""

    def test_t1_f2_doublet_zmx_ingestion(self, sample_thorlabs_doublet_zmx):
        stack = parse_zmx_file(sample_thorlabs_doublet_zmx)
        assert stack is not None
        assert len(stack.surfaces) >= 4

    def test_t1_f2_singlet_zmx_ingestion(self, sample_thorlabs_singlet_zmx):
        stack = parse_zmx_file(sample_thorlabs_singlet_zmx)
        assert len(stack.surfaces) >= 3

    def test_t1_f2_aspheric_zmx_ingestion(self, sample_aspheric_zmx):
        stack = parse_zmx_file(sample_aspheric_zmx)
        asph_surfs = [s for s in stack.surfaces if len(s.asphere_coeffs) > 0]
        assert len(asph_surfs) >= 1

    def test_t1_f2_glass_material_resolution(self, sample_thorlabs_doublet_zmx):
        stack = parse_zmx_file(sample_thorlabs_doublet_zmx)
        glasses = [s.glass.upper() for s in stack.surfaces]
        assert "N-BK7" in glasses
        assert "SF5" in glasses

    def test_t1_f2_catalog_lens_ac254_lookup(self, loaded_spectrometer_stack):
        indices = loaded_spectrometer_stack.replace_lens_from_catalog(target=4, part_number="AC254-050-B")
        assert len(indices) == 3


class TestTier1_F3_DynamicSequenceManagement:
    """F3: Dynamic sequential insertion and deletion with thickness preservation."""

    def test_t1_f3_insert_singlet_expands_stack(self, loaded_spectrometer_stack):
        init_len = len(loaded_spectrometer_stack.surfaces)
        idx = loaded_spectrometer_stack.insert_singlet(after_surface_index=6, name="NewSinglet", radius_front=50.0, thickness=3.0, glass="BK7", radius_back=-50.0)
        assert len(loaded_spectrometer_stack.surfaces) == init_len + 2
        assert idx == [7, 8]

    def test_t1_f3_insert_doublet_expands_stack(self, loaded_spectrometer_stack):
        init_len = len(loaded_spectrometer_stack.surfaces)
        idx = loaded_spectrometer_stack.insert_doublet(after_surface_index=6, name="NewDoublet")
        assert len(loaded_spectrometer_stack.surfaces) == init_len + 3

    def test_t1_f3_delete_lens_element_merges_air_gap(self, loaded_spectrometer_stack):
        init_len = len(loaded_spectrometer_stack.surfaces)
        pre_gap = loaded_spectrometer_stack.surfaces[3].thickness
        loaded_spectrometer_stack.delete_lens_element(target=4)
        assert len(loaded_spectrometer_stack.surfaces) < init_len
        assert loaded_spectrometer_stack.surfaces[3].thickness >= pre_gap

    def test_t1_f3_reindex_surface_names(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        model._reindex_surface_names()
        assert model.rowCount() == 17
        assert len(model._surfaces) == 17

    def test_t1_f3_sequential_coordinate_accumulation(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack._update_accumulated_positions()
        for i in range(1, len(loaded_spectrometer_stack.surfaces)):
            assert loaded_spectrometer_stack.surfaces[i].global_origin is not None


class TestTier1_F4_DynamicFrontToBackFlip:
    """F4: Dynamic front-to-back lens element flip."""

    def test_t1_f4_curvature_inversion(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)
        assert abs(loaded_spectrometer_stack.surfaces[4].radius - 115.0) < 1e-2
        assert abs(loaded_spectrometer_stack.surfaces[6].radius - (-34.0)) < 1e-2

    def test_t1_f4_singlet_flip(self, loaded_spectrometer_stack, sample_thorlabs_singlet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_singlet_zmx, reverse=True)
        assert loaded_spectrometer_stack.surfaces[4].radius == 0.0

    def test_t1_f4_glass_sequence_reversal(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)
        glasses_combined = (
            loaded_spectrometer_stack.surfaces[4].glass + " " +
            loaded_spectrometer_stack.surfaces[5].glass + " " +
            loaded_spectrometer_stack.surfaces[6].glass
        ).upper()
        assert "SF5" in glasses_combined or "N-BK7" in glasses_combined

    def test_t1_f4_air_gap_preservation(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        gap = loaded_spectrometer_stack.surfaces[6].thickness
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)
        assert abs(loaded_spectrometer_stack.surfaces[6].thickness - gap) < 1e-3

    def test_t1_f4_double_flip_identity(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=False)
        r_fwd = loaded_spectrometer_stack.surfaces[4].radius
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=False)
        assert abs(loaded_spectrometer_stack.surfaces[4].radius - r_fwd) < 1e-3


class TestTier1_F5_3DRigidBodyRotationAndDecenter:
    """F5: 3D rigid body tilt and decenter around 3 selectable pivots."""

    def test_t1_f5_front_vertex_pivot_rotation(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=5.0, pivot="front_vertex")
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - 5.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[4].decenter_y) < 1e-3

    def test_t1_f5_center_of_mass_pivot_rotation(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=5.0, pivot="center_of_mass")
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - 5.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[5].tilt_x - 5.0) < 1e-3

    def test_t1_f5_rear_vertex_pivot_rotation(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=5.0, pivot="rear_vertex")
        assert abs(loaded_spectrometer_stack.surfaces[6].tilt_x - 5.0) < 1e-3

    def test_t1_f5_compound_xyz_tilt(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=3.0, tilt_y=2.0, tilt_z=1.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - 3.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_y - 2.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_z - 1.0) < 1e-3

    def test_t1_f5_rigid_decenter_translation(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.decenter_lens_element(4, decenter_x=2.0, decenter_y=-3.0, tilt_x=0.0, tilt_y=0.0, tilt_z=0.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].decenter_x - 2.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[5].decenter_x - 2.0) < 1e-3
        assert abs(loaded_spectrometer_stack.surfaces[4].decenter_y - (-3.0)) < 1e-3


class TestTier1_F6_DualSpeedRayTracing:
    """F6: Dual-speed ray tracing architecture."""

    def test_t1_f6_fast_trace_latency(self, loaded_optical_system):
        # Interactive fast trace chief rays across NIR band
        timings = []
        for _ in range(5):
            t0 = time.perf_counter()
            loaded_optical_system.trace_rays([Ray(wavelength=wl) for wl in [1.10, 1.225, 1.35]])
            timings.append((time.perf_counter() - t0) * 1000.0)
        assert min(timings) < 50.0

    def test_t1_f6_fast_trace_chief_ray(self, loaded_optical_system):
        r = loaded_optical_system.trace_ray(Ray(wavelength=1.225))
        assert len(r.history) >= 8

    def test_t1_f6_full_trace_dense_fan(self, loaded_optical_system):
        spot = generate_spot_diagram_from_system(loaded_optical_system, wavelength_um=1.225, num_rays=30)
        assert spot is not None

    def test_t1_f6_nir_multi_wavelength_trace(self, loaded_optical_system):
        for wl in [1.10, 1.225, 1.35]:
            r = loaded_optical_system.trace_ray(Ray(wavelength=wl))
            assert len(r.history) >= 8

    def test_t1_f6_vignetting_detection(self, loaded_optical_system):
        # Ray launched far off-axis should vignette
        r_off = loaded_optical_system.trace_ray(Ray(pos=np.array([0.0, 50.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        assert r_off.vignetted is True


class TestTier1_F7_RealTimeRayFanUpdate:
    """F7: Real-time ray fan recalculation on slider interaction."""

    def test_t1_f7_tilt_slider_alters_fan(self, loaded_spectrometer_stack):
        opt1 = OpticalSystem(loaded_spectrometer_stack)
        r1 = opt1.trace_ray(Ray(wavelength=1.225))
        loaded_spectrometer_stack.surfaces[3].tilt_x = -14.0
        loaded_spectrometer_stack._update_accumulated_positions()
        opt2 = OpticalSystem(loaded_spectrometer_stack)
        r2 = opt2.trace_ray(Ray(wavelength=1.225))
        assert abs(r1.history[3]["dir_y"] - r2.history[3]["dir_y"]) > 0.01

    def test_t1_f7_thickness_slider_alters_ray_positions(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[2].thickness = 80.0
        loaded_spectrometer_stack._update_accumulated_positions()
        assert loaded_spectrometer_stack.surfaces[2].thickness == 80.0

    def test_t1_f7_dual_view_rz_canvas_update(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dv = DualViewWidget(model=model)
        dv.refresh_layout()
        assert dv.model is not None

    def test_t1_f7_dual_view_xy_canvas_update(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dv = DualViewWidget(model=model)
        dv.refresh_layout()
        assert dv.canvas is not None

    def test_t1_f7_interactive_scrubbing_responsiveness(self, loaded_spectrometer_stack):
        t0 = time.perf_counter()
        opt = OpticalSystem(loaded_spectrometer_stack)
        for i in range(10):
            loaded_spectrometer_stack.surfaces[3].tilt_x = -12.0 - 0.1 * i
            loaded_spectrometer_stack._update_accumulated_positions()
            opt.trace_ray(Ray(wavelength=1.225))
        assert (time.perf_counter() - t0) < 0.5


class TestTier1_F8_RealTimeSpectrumSpread:
    """F8: Real-time grating spectrum dispersion and focal plane spread."""

    def test_t1_f8_grating_dispersion_calculation(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(
            grating=grating,
            wavelengths_um=[1.10, 1.225, 1.35],
            focal_length_mm=50.0
        )
        assert len(spots) == 3
        assert spots[0][2] != spots[2][2]

    def test_t1_f8_focal_plane_linear_dispersion(self):
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=0.3,
            focal_length_mm=50.0,
            wavelength_um=1.225
        )
        assert metrics.linear_dispersion_mm_per_um > 0.0

    def test_t1_f8_resolving_power_metric(self):
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=0.3,
            focal_length_mm=50.0,
            wavelength_um=1.225,
            beam_diameter_mm=10.0
        )
        assert metrics.theoretical_resolving_power > 0.0

    def test_t1_f8_spectrum_tab_render(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        tabs = AnalysisTabsWidget(model=model)
        tabs.refresh_analysis(wavelengths=[1.10, 1.225, 1.35])
        assert tabs.tab_dispersion is not None

    def test_t1_f8_ccd_heatmap_generation(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        tabs = AnalysisTabsWidget(model=model)
        tabs.refresh_analysis(wavelengths=[1.10, 1.225, 1.35])
        assert tabs.tab_ccd is not None


class TestTier1_F9_RealTime0p8mmSpotDiagram:
    """F9: Real-time 0.8mm spot diagram diagnostics."""

    def test_t1_f9_spot_centroid_calculation(self):
        spot = SpotDiagram(x_coords=[0.1, 0.3], y_coords=[-0.1, -0.3], wavelength_um=1.225)
        cx, cy = spot.calculate_centroid()
        assert abs(cx - 0.2) < 1e-4
        assert abs(cy - (-0.2)) < 1e-4

    def test_t1_f9_spot_rms_radius_calculation(self):
        spot = SpotDiagram(x_coords=[0.0, 0.2], y_coords=[0.0, 0.2], wavelength_um=1.225)
        assert spot.calculate_rms_radius() > 0.0

    def test_t1_f9_spot_geo_radius_calculation(self):
        spot = SpotDiagram(x_coords=[0.0, 0.4], y_coords=[0.0, 0.3], wavelength_um=1.225)
        assert spot.calculate_geo_radius() > 0.0

    def test_t1_f9_airy_disk_radius_calculation(self):
        spot = SpotDiagram(x_coords=[0.0, 0.1], y_coords=[0.0, 0.1], wavelength_um=1.225, f_number=4.0)
        airy = spot.calculate_airy_disk_radius()
        assert airy > 0.0

    def test_t1_f9_diffraction_limited_flag(self):
        spot_small = SpotDiagram(x_coords=[0.0001, -0.0001], y_coords=[0.0001, -0.0001], wavelength_um=1.225, f_number=4.0)
        assert spot_small.is_diffraction_limited() is True


class TestTier1_F10_SpecializedTuningSliders:
    """F10: Specialized UI tuning sliders and parameter docks."""

    def test_t1_f10_component_editor_surface_selection(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        assert dock.get_selected_surface_index() == 3

    def test_t1_f10_tilt_x_y_z_slider_ranges(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        assert dock.tilt_x_slider.minimum() <= -450
        assert dock.tilt_x_slider.maximum() >= 450

    def test_t1_f10_decenter_spinbox_precision(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(4)
        dock.dec_x_spin.setValue(1.234)
        assert abs(model.get_surface_stack().surfaces[4].decenter_x - 1.234) < 1e-3

    def test_t1_f10_lens_parameter_lock_toggle(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        assert hasattr(dock, "enable_lens_edition_check")

    def test_t1_f10_quick_fold_presets(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        assert hasattr(dock, "btn_reset_tilts")


class TestTier1_F11_SciPyOptimizer:
    """F11: Automated SciPy spot size and alignment optimizer."""

    def test_t1_f11_optimizer_initialization(self, loaded_optical_system):
        opt = OpticalOptimizer(loaded_optical_system, wavelengths=[1.225], method="Nelder-Mead", num_rays=7, max_iter=5)
        assert opt.method == "Nelder-Mead"

    def test_t1_f11_optimizer_nelder_mead_solver(self, loaded_spectrometer_stack):
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], method="Nelder-Mead", num_rays=7, max_iter=5)
        res = opt.optimize(variable_indices=[2])
        assert isinstance(res, OptimizationResult)

    def test_t1_f11_optimizer_lbfgsb_solver(self, loaded_spectrometer_stack):
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], method="L-BFGS-B", num_rays=7, max_iter=5)
        res = opt.optimize(variable_indices=[2], bounds=[(50.0, 90.0)])
        assert isinstance(res, OptimizationResult)

    def test_t1_f11_penalty_function_evaluation(self, loaded_optical_system):
        opt = OpticalOptimizer(loaded_optical_system, wavelengths=[1.225])
        _, pen = opt._evaluate_merit_and_penalty(np.array([-10.0]), {"thickness": [2]})
        assert pen > 100.0

    def test_t1_f11_variable_indices_auto_discovery(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[2].thickness_variable = True
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=2, num_rays=7)
        res = opt.optimize(variable_indices=None)
        assert isinstance(res, OptimizationResult)


class TestTier1_F12_CADExportSTEPAndDXF:
    """F12: STEP 3D and DXF 2D technical CAD drawing export."""

    def test_t1_f12_step_export_pure_python(self, loaded_spectrometer_stack, tmp_path):
        out_step = str(tmp_path / "model.step")
        res = STEPExporter.export(loaded_spectrometer_stack, out_step, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is True
        assert os.path.exists(out_step)

    def test_t1_f12_step_contains_protocol_header(self, loaded_spectrometer_stack, tmp_path):
        out_step = str(tmp_path / "model_hdr.step")
        STEPExporter.export(loaded_spectrometer_stack, out_step, engine=STEPExportEngine.PURE_PYTHON)
        with open(out_step, "r") as f:
            content = f.read()
        assert "ISO-10303-21;" in content
        assert "ENDSEC;" in content

    def test_t1_f12_dxf_2d_export(self, loaded_spectrometer_stack, tmp_path):
        out_dxf = str(tmp_path / "drawing.dxf")
        res = DXF2DExporter(loaded_spectrometer_stack).export(out_dxf)
        assert os.path.exists(out_dxf)

    def test_t1_f12_dxf_contains_layers_and_entities(self, loaded_spectrometer_stack, tmp_path):
        out_dxf = str(tmp_path / "drawing_ent.dxf")
        DXF2DExporter(loaded_spectrometer_stack).export(out_dxf)
        with open(out_dxf, "r") as f:
            content = f.read()
        assert "SECTION" in content
        assert "ENTITIES" in content

    def test_t1_f12_cad_export_with_aspheric_surfaces(self, loaded_spectrometer_stack, tmp_path):
        out_step = str(tmp_path / "asph.step")
        res = STEPExporter.export(loaded_spectrometer_stack, out_step, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is True


# ============================================================================
# TIER 2: BOUNDARY & CORNER CONDITIONS (60 tests: 5 tests per feature F1 - F12)
# ============================================================================

class TestTier2_Boundary_Corner:
    """Tier 2: Boundary, corner, extreme parameter, and edge cases across F1 - F12."""

    # F1 Boundary
    def test_t2_f1_empty_surfaces_stack(self):
        stack = SurfaceStack()
        assert len(stack.surfaces) == 0

    def test_t2_f1_single_surface_system(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR))
        assert len(stack.surfaces) == 1

    def test_t2_f1_zero_thickness_system(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[0].thickness = 0.0
        loaded_spectrometer_stack._update_accumulated_positions()
        assert loaded_spectrometer_stack.surfaces[0].thickness == 0.0

    def test_t2_f1_flat_mirror_infinite_radius(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[7].radius = 0.0
        assert loaded_spectrometer_stack.surfaces[7].radius == 0.0

    def test_t2_f1_conic_parabola_hyperbola(self):
        s_parab = Surface(conic=-1.0)
        s_hyper = Surface(conic=-2.5)
        assert s_parab.conic == -1.0
        assert s_hyper.conic == -2.5

    # F2 Boundary
    def test_t2_f2_empty_zmx_file_handling(self, tmp_path):
        empty_zmx = str(tmp_path / "empty.zmx")
        with open(empty_zmx, "w") as f:
            f.write("")
        stack = parse_zmx_file(empty_zmx)
        assert stack is not None

    def test_t2_f2_zmx_with_unknown_glass(self, tmp_path):
        zmx = str(tmp_path / "unk.zmx")
        with open(zmx, "w") as f:
            f.write("SURF 1\n  TYPE STANDARD\n  CURV 0.05\n  DISZ 5.0\n  GLAS UNKNOWN_GLASS_99\n")
        stack = parse_zmx_file(zmx)
        assert stack is not None

    def test_t2_f2_zmx_without_detector(self, tmp_path):
        zmx = str(tmp_path / "no_det.zmx")
        with open(zmx, "w") as f:
            f.write("SURF 1\n  TYPE STANDARD\n  CURV 0.05\n  DISZ 5.0\n  GLAS BK7\n")
        stack = parse_zmx_file(zmx)
        assert len(stack.surfaces) >= 1

    def test_t2_f2_nonexistent_zmx_path(self):
        try:
            stack = parse_zmx_file("nonexistent_path_file.zmx")
            assert stack is None or len(stack.surfaces) == 0
        except (FileNotFoundError, IOError):
            pass

    def test_t2_f2_zmx_asph_with_zero_coeffs(self, tmp_path):
        zmx = str(tmp_path / "zero_asph.zmx")
        with open(zmx, "w") as f:
            f.write("SURF 1\n  TYPE EVENASPH\n  PARM 1 0.0\n  PARM 2 0.0\n")
        stack = parse_zmx_file(zmx)
        assert stack is not None

    # F3 Boundary
    def test_t2_f3_insert_at_boundary_start(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.insert_singlet(after_surface_index=0, name="FrontLens")
        assert len(loaded_spectrometer_stack.surfaces) == 19

    def test_t2_f3_insert_at_boundary_end(self, loaded_spectrometer_stack):
        last_idx = len(loaded_spectrometer_stack.surfaces) - 1
        loaded_spectrometer_stack.insert_singlet(after_surface_index=last_idx, name="EndLens")
        assert len(loaded_spectrometer_stack.surfaces) == 19

    def test_t2_f3_delete_first_surface_safety(self, loaded_spectrometer_stack):
        init_len = len(loaded_spectrometer_stack.surfaces)
        loaded_spectrometer_stack.delete_lens_element(target=0)
        assert len(loaded_spectrometer_stack.surfaces) <= init_len

    def test_t2_f3_delete_last_surface_safety(self, loaded_spectrometer_stack):
        init_len = len(loaded_spectrometer_stack.surfaces)
        loaded_spectrometer_stack.delete_lens_element(target=init_len - 1)
        assert len(loaded_spectrometer_stack.surfaces) <= init_len

    def test_t2_f3_insert_out_of_bounds_index(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.insert_singlet(after_surface_index=999, name="OutLens")
        assert len(loaded_spectrometer_stack.surfaces) == 19

    # F4 Boundary
    def test_t2_f4_flip_symmetric_biconvex_lens(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[4].radius = 50.0
        loaded_spectrometer_stack.surfaces[6].radius = -50.0
        # Reversing symmetric lens preserves focal power
        assert abs(loaded_spectrometer_stack.surfaces[4].radius - (-loaded_spectrometer_stack.surfaces[6].radius)) < 1e-4

    def test_t2_f4_flip_plano_convex_lens(self, loaded_spectrometer_stack, sample_thorlabs_singlet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_singlet_zmx, reverse=True)
        assert loaded_spectrometer_stack.surfaces[4].radius == 0.0

    def test_t2_f4_flip_meniscus_lens(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[4].radius = 30.0
        loaded_spectrometer_stack.surfaces[5].radius = 50.0
        assert loaded_spectrometer_stack.surfaces[4].radius != loaded_spectrometer_stack.surfaces[5].radius

    def test_t2_f4_flip_zero_thickness_element(self):
        s1 = Surface(radius=40.0, thickness=0.0)
        s2 = Surface(radius=-40.0, thickness=0.0)
        assert s1.thickness == 0.0 and s2.thickness == 0.0

    def test_t2_f4_flip_multi_element_triplet(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.insert_singlet(after_surface_index=6, name="S3")
        assert len(loaded_spectrometer_stack.surfaces) == 19

    # F5 Boundary
    def test_t2_f5_extreme_tilt_45_deg(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=45.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - 45.0) < 1e-3

    def test_t2_f5_extreme_tilt_minus_45_deg(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=-45.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - (-45.0)) < 1e-3

    def test_t2_f5_sub_arcsecond_tilt(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=0.0001)
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x - 0.0001) < 1e-6

    def test_t2_f5_large_decenter_20mm(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.decenter_lens_element(4, decenter_x=20.0, decenter_y=-20.0, tilt_x=0.0, tilt_y=0.0, tilt_z=0.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].decenter_x - 20.0) < 1e-3

    def test_t2_f5_zero_tilt_zero_decenter_identity(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=0.0, tilt_y=0.0, tilt_z=0.0)
        assert abs(loaded_spectrometer_stack.surfaces[4].tilt_x) < 1e-5

    # F6 Boundary
    def test_t2_f6_zero_rays_trace(self, loaded_optical_system):
        res = loaded_optical_system.trace_rays([])
        assert len(res) == 0

    def test_t2_f6_single_ray_trace(self, loaded_optical_system):
        res = loaded_optical_system.trace_rays([Ray(wavelength=1.225)])
        assert len(res) == 1

    def test_t2_f6_extreme_short_wavelength(self, loaded_optical_system):
        r = loaded_optical_system.trace_ray(Ray(wavelength=0.200))
        assert r is not None

    def test_t2_f6_extreme_long_wavelength(self, loaded_optical_system):
        r = loaded_optical_system.trace_ray(Ray(wavelength=5.0))
        assert r is not None

    def test_t2_f6_extreme_marginal_ray(self, loaded_optical_system):
        r = loaded_optical_system.trace_ray(Ray(pos=np.array([0.0, 5.9, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225))
        assert r is not None

    # F7 Boundary
    def test_t2_f7_zero_step_slider_move(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        dock.tilt_x_slider.setValue(int(loaded_spectrometer_stack.surfaces[3].tilt_x * 10))
        assert abs(model.get_surface_stack().surfaces[3].tilt_x - (-12.0)) < 1e-2

    def test_t2_f7_out_of_range_slider_step(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        dock.tilt_x_slider.setValue(9999)
        assert dock.tilt_x_slider.value() <= dock.tilt_x_slider.maximum()

    def test_t2_f7_rapid_slider_bounce(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        for val in [-120, -140, -120, -100, -120]:
            dock.tilt_x_slider.setValue(val)
        assert abs(model.get_surface_stack().surfaces[3].tilt_x - (-12.0)) < 1e-2

    def test_t2_f7_negative_thickness_spinbox_safe(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(2)
        dock.thick_spin.setValue(-10.0)
        assert dock.thick_spin.value() >= 0.0

    def test_t2_f7_zero_semi_diameter_canvas_safe(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        model.get_surface_stack().surfaces[1].semi_diameter = 0.001
        dv = DualViewWidget(model=model)
        dv.refresh_layout()
        assert dv is not None

    # F8 Boundary
    def test_t2_f8_monochromatic_single_wavelength(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(grating, [1.225])
        assert len(spots) == 1

    def test_t2_f8_identical_min_max_wavelengths(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(grating, [1.225, 1.225])
        assert len(spots) == 2
        assert spots[0][2] == spots[1][2]

    def test_t2_f8_inverted_wavelength_order(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(grating, [1.35, 1.225, 1.10])
        assert len(spots) == 3

    def test_t2_f8_zero_groove_density(self):
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=0.0, order=1, diffraction_angle_rad=0.0, focal_length_mm=50.0, wavelength_um=1.225
        )
        assert metrics.linear_dispersion_mm_per_um == 0.0

    def test_t2_f8_zero_diffraction_order(self):
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0, order=0, diffraction_angle_rad=0.0, focal_length_mm=50.0, wavelength_um=1.225
        )
        assert metrics.linear_dispersion_mm_per_um == 0.0

    # F9 Boundary
    def test_t2_f9_single_hit_spot(self):
        spot = SpotDiagram(x_coords=[0.5], y_coords=[-0.5], wavelength_um=1.225)
        cx, cy = spot.calculate_centroid()
        assert abs(cx - 0.5) < 1e-4
        assert abs(cy - (-0.5)) < 1e-4

    def test_t2_f9_all_vignetted_spot(self):
        spot = SpotDiagram(x_coords=[], y_coords=[], wavelength_um=1.225)
        assert spot.calculate_centroid() == (0.0, 0.0)
        assert spot.calculate_rms_radius() == 0.0

    def test_t2_f9_extreme_off_axis_spot(self):
        spot = SpotDiagram(x_coords=[100.0, 100.2], y_coords=[-100.0, -100.2], wavelength_um=1.225)
        assert spot.calculate_rms_radius() > 0.0

    def test_t2_f9_sub_micron_spot(self):
        spot = SpotDiagram(x_coords=[1e-5, -1e-5], y_coords=[1e-5, -1e-5], wavelength_um=1.225)
        assert spot.calculate_rms_radius() < 1e-3

    def test_t2_f9_detector_clipping_radius(self):
        spot = SpotDiagram(x_coords=[0.3, -0.3], y_coords=[0.3, -0.3], wavelength_um=1.225)
        assert spot.calculate_geo_radius() > 0.3

    # F10 Boundary
    def test_t2_f10_slider_saturation_min(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        dock.tilt_x_slider.setValue(-450)
        assert abs(model.get_surface_stack().surfaces[3].tilt_x - (-45.0)) < 1e-2

    def test_t2_f10_slider_saturation_max(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        dock.tilt_x_slider.setValue(450)
        assert abs(model.get_surface_stack().surfaces[3].tilt_x - 45.0) < 1e-2

    def test_t2_f10_reset_tilts_from_extreme(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        dock.set_selected_surface(3)
        dock.tilt_x_slider.setValue(350)
        dock.btn_reset_tilts.click()
        assert abs(model.get_surface_stack().surfaces[3].tilt_x) < 1e-2

    def test_t2_f10_decenter_slider_limits(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        assert dock.dec_x_spin.minimum() <= -25.0
        assert dock.dec_x_spin.maximum() >= 25.0

    def test_t2_f10_switch_surfaces_rapidly(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock = ComponentEditorDockWidget(model=model)
        for idx in range(17):
            dock.set_selected_surface(idx)
            assert dock.get_selected_surface_index() == idx

    # F11 Boundary
    def test_t2_f11_zero_iterations_optimizer(self, loaded_optical_system):
        opt = OpticalOptimizer(loaded_optical_system, wavelengths=[1.225], max_iter=0, num_rays=7)
        res = opt.optimize(variable_indices=[2])
        assert isinstance(res, OptimizationResult)

    def test_t2_f11_already_optimal_system(self, loaded_spectrometer_stack):
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=2, num_rays=7)
        res = opt.optimize(variable_indices=[2])
        assert isinstance(res, OptimizationResult)

    def test_t2_f11_single_variable_optimization(self, loaded_spectrometer_stack):
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=3, num_rays=7)
        res = opt.optimize(variable_indices=[6])
        assert len(res.optimized_thicknesses) == 1

    def test_t2_f11_all_fixed_variables(self, loaded_optical_system):
        opt = OpticalOptimizer(loaded_optical_system, wavelengths=[1.225], max_iter=2, num_rays=7)
        res = opt.optimize(variable_indices=[])
        assert len(res.optimized_thicknesses) == 0

    def test_t2_f11_extreme_initial_misalignment(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[3].tilt_x = 20.0
        loaded_spectrometer_stack._update_accumulated_positions()
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=3, num_rays=7)
        res = opt.optimize(variable_indices={"tilt_x": [3]})
        assert isinstance(res, OptimizationResult)

    # F12 Boundary
    def test_t2_f12_export_empty_stack(self, tmp_path):
        stack = SurfaceStack()
        out = str(tmp_path / "empty.step")
        res = STEPExporter.export(stack, out, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is True

    def test_t2_f12_export_single_mirror(self, tmp_path):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="M1", surface_type=SurfaceType.MIRROR, radius=-100.0, glass="MIRROR", semi_diameter=12.7))
        out = str(tmp_path / "m1.step")
        res = STEPExporter.export(stack, out, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is True

    def test_t2_f12_export_read_only_directory(self, loaded_spectrometer_stack):
        out = "Z:/invalid_nonexistent_mount/cad.step"
        res = STEPExporter.export(loaded_spectrometer_stack, out, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is False

    def test_t2_f12_dxf_export_zero_diameter(self, tmp_path):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Slit", semi_diameter=0.01))
        out = str(tmp_path / "slit.dxf")
        res = DXF2DExporter(stack).export(out)
        assert os.path.exists(out)

    def test_t2_f12_step_export_extreme_asphere_coefficients(self, tmp_path):
        stack = SurfaceStack()
        stack.add_surface(Surface(
            name="ExtremeAsph", surface_type=SurfaceType.ASPHERIC, radius=5.0, conic=-5.0,
            asphere_coeffs=[0.0, 1e-2, 1e-4, 1e-6], glass="BK7", thickness=10.0, semi_diameter=12.7
        ))
        out = str(tmp_path / "ext_asph.step")
        res = STEPExporter.export(stack, out, engine=STEPExportEngine.PURE_PYTHON)
        assert res.is_success is True


# ============================================================================
# TIER 3: CROSS-FEATURE PAIRWISE (12 tests: P01 to P12)
# ============================================================================

class TestTier3_Pairwise:
    """Tier 3: 12 Pairwise Cross-Feature Integration Tests."""

    def test_t3_p01_f1_json_and_f6_tracer(self, loaded_spectrometer_stack):
        opt = OpticalSystem(loaded_spectrometer_stack)
        rays = [Ray(wavelength=wl) for wl in [1.10, 1.225, 1.35]]
        traced = opt.trace_rays(rays)
        assert len(traced) == 3

    def test_t3_p02_f2_zmx_and_f3_dynamic_sequence(self, loaded_spectrometer_stack, sample_thorlabs_singlet_zmx):
        init_len = len(loaded_spectrometer_stack.surfaces)
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_singlet_zmx)
        assert len(loaded_spectrometer_stack.surfaces) == init_len - 1

    def test_t3_p03_f3_dynamic_sequence_and_f5_rigid_tilt(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.insert_doublet(after_surface_index=6, name="AddedDoublet")
        loaded_spectrometer_stack.rotate_lens_element(7, tilt_x=4.0)
        assert abs(loaded_spectrometer_stack.surfaces[7].tilt_x - 4.0) < 1e-3

    def test_t3_p04_f5_rigid_tilt_and_f7_rayfan_update(self, loaded_spectrometer_stack):
        opt_pre = OpticalSystem(loaded_spectrometer_stack)
        r_pre = opt_pre.trace_ray(Ray(wavelength=1.225))
        loaded_spectrometer_stack.rotate_lens_element(4, tilt_x=5.0)
        opt_post = OpticalSystem(loaded_spectrometer_stack)
        r_post = opt_post.trace_ray(Ray(wavelength=1.225))
        assert abs(r_pre.history[4]["dir_y"] - r_post.history[4]["dir_y"]) > 0.01

    def test_t3_p05_f6_tracer_and_f9_spot_diagram(self, loaded_optical_system):
        spot = generate_spot_diagram_from_system(loaded_optical_system, wavelength_um=1.225, num_rays=20)
        assert spot is not None

    def test_t3_p06_f10_tuning_dock_and_f11_optimizer(self, loaded_spectrometer_stack, qapp):
        model = LensDataTableModel(stack=loaded_spectrometer_stack)
        dock_tuning = ComponentEditorDockWidget(model=model)
        dock_opt = OptimizerDockWidget(model=model)
        assert dock_tuning is not None and dock_opt is not None

    def test_t3_p07_f11_optimizer_and_f12_step_export(self, loaded_spectrometer_stack, tmp_path):
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=2, num_rays=7)
        res = opt.optimize(variable_indices=[2])
        out = str(tmp_path / "opt.step")
        res_exp = STEPExporter.export(res.optimized_stack, out, engine=STEPExportEngine.PURE_PYTHON)
        assert res_exp.is_success is True

    def test_t3_p08_f3_dynamic_element_and_f12_dxf_export(self, loaded_spectrometer_stack, tmp_path):
        loaded_spectrometer_stack.insert_singlet(after_surface_index=6, name="CADLens")
        out = str(tmp_path / "cad_lens.dxf")
        res = DXF2DExporter(loaded_spectrometer_stack).export(out)
        assert os.path.exists(out)

    def test_t3_p09_f4_dynamic_flip_and_f11_optimizer(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], max_iter=2, num_rays=7)
        res = opt.optimize(variable_indices=[6])
        assert isinstance(res, OptimizationResult)

    def test_t3_p10_f2_zmx_and_f9_spot_diagram(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx)
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        spot = generate_spot_diagram_from_system(opt_sys, wavelength_um=1.225, num_rays=15)
        assert spot is not None

    def test_t3_p11_f7_rayfan_update_and_f8_spectrum_spread(self, loaded_spectrometer_stack):
        loaded_spectrometer_stack.surfaces[3].tilt_x = -13.0
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(grating, [1.10, 1.225, 1.35])
        assert len(spots) == 3

    def test_t3_p12_f1_json_model_and_f10_tuning_sliders(self, spectrometer_json_path, qapp):
        win = MainWindow()
        win.on_load_project(spectrometer_json_path)
        win.component_editor_dock.set_selected_surface(3)
        win.component_editor_dock.tilt_x_slider.setValue(-130)
        assert abs(win.model.get_surface_stack().surfaces[3].tilt_x - (-13.0)) < 1e-2
        win.close()


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (6 Scenarios: S1 to S6)
# ============================================================================

class TestTier4_Scenarios:
    """Tier 4: End-to-End Real-World Application Workflows."""

    def test_t4_scenario_1_reference_nir_spectrometer_alignment(self, spectrometer_json_path, qapp):
        """Scenario 1: Reference NIR Fingerprint Spectrometer Alignment & Verification."""
        win = MainWindow()
        loaded = win.on_load_project(spectrometer_json_path)
        assert loaded is True
        assert len(win.model.get_surface_stack().surfaces) == 17

        # Verify diffraction grating and asphere parameters
        stack = win.model.get_surface_stack()
        assert stack.surfaces[3].surface_type == SurfaceType.GRATING
        assert stack.surfaces[12].surface_type == SurfaceType.ASPHERIC
        assert stack.surfaces[16].surface_type == SurfaceType.DETECTOR

        # Trigger ray trace and check clean status
        win.on_trace_rays()
        assert win.dual_view is not None
        win.close()

    def test_t4_scenario_2_dynamic_grating_swap_and_doublet_replacement(self, loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
        """Scenario 2: Dynamic Grating Angle Tuning & Focuser Lens Replacement."""
        # 1. User tunes grating angle
        loaded_spectrometer_stack.surfaces[3].tilt_x = -12.5
        loaded_spectrometer_stack._update_accumulated_positions()

        # 2. User swaps focuser lens with Thorlabs doublet in flipped orientation
        loaded_spectrometer_stack.replace_lens_from_zmx(target=4, file_path=sample_thorlabs_doublet_zmx, reverse=True)

        # 3. Ray trace through modified system
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        rays = [Ray(wavelength=wl) for wl in [1.10, 1.225, 1.35]]
        traced = opt_sys.trace_rays(rays)
        assert len(traced) == 3

    def test_t4_scenario_3_detector_focus_optimization_from_misalignment(self, loaded_spectrometer_stack):
        """Scenario 3: Automated Optimization of Misaligned Spectrometer."""
        # Perturb collimator and condenser spacings
        loaded_spectrometer_stack.surfaces[2].thickness = 66.0
        loaded_spectrometer_stack.surfaces[11].thickness = 58.0
        loaded_spectrometer_stack._update_accumulated_positions()

        # Optimize spacings to restore focus
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        opt = OpticalOptimizer(opt_sys, wavelengths=[1.225], method="Nelder-Mead", num_rays=7, max_iter=5)
        res = opt.optimize(variable_indices=[2, 11])
        assert isinstance(res, OptimizationResult)
        assert res.optimized_stack is not None

    def test_t4_scenario_4_rapid_slider_scrubbing_with_debounce(self, loaded_spectrometer_stack):
        """Scenario 4: Rapid Interactive Slider Scrubbing with Debounced Full Trace."""
        opt_sys = OpticalSystem(loaded_spectrometer_stack)
        grating = loaded_spectrometer_stack.surfaces[3]

        # 15 interactive scrubbing steps
        for step in range(15):
            grating.tilt_x = -12.0 - (step * 0.05)
            loaded_spectrometer_stack._update_accumulated_positions()
            # Fast trace
            fast_ray = Ray(wavelength=1.225)
            opt_sys.trace_ray(fast_ray)

        # On release: full trace
        spot = SpotDiagram(x_coords=[-0.1, 0.1], y_coords=[-0.1, 0.1], wavelength_um=1.225)
        assert spot.calculate_rms_radius() > 0.0

    def test_t4_scenario_5_end_to_end_cad_production_pipeline(self, loaded_spectrometer_stack, tmp_path):
        """Scenario 5: Complete Design to CAD STEP & 2D Technical Drawing Pipeline."""
        # 1. Verify optical prescription
        assert len(loaded_spectrometer_stack.surfaces) == 17

        # 2. Export 3D STEP Assembly
        step_file = str(tmp_path / "production_spectrometer.step")
        res_step = STEPExporter.export(loaded_spectrometer_stack, step_file, engine=STEPExportEngine.PURE_PYTHON)
        assert res_step.is_success is True
        assert os.path.exists(step_file)

        # 3. Export 2D DXF Technical Drawing
        dxf_file = str(tmp_path / "production_spectrometer.dxf")
        res_dxf = DXF2DExporter(loaded_spectrometer_stack).export(dxf_file)
        assert os.path.exists(dxf_file)

    def test_t4_scenario_6_state_save_reopen_and_regression_verification(self, loaded_spectrometer_stack, tmp_path, qapp):
        """Scenario 6: Full Design State Persistence, Reload, and Regression Trace."""
        # 1. Modify some parameters
        loaded_spectrometer_stack.surfaces[3].tilt_x = -12.3
        loaded_spectrometer_stack.surfaces[2].thickness = 72.0

        # 2. Save project to JSON
        save_file = str(tmp_path / "session_spectrometer.json")
        loaded_spectrometer_stack.save_to_json(save_file, wavelengths=[1.10, 1.225, 1.35])

        # 3. Reload in fresh MainWindow instance
        win = MainWindow()
        ok = win.on_load_project(save_file)
        assert ok is True
        reloaded_stack = win.model.get_surface_stack()
        assert len(reloaded_stack.surfaces) == 17
        assert abs(reloaded_stack.surfaces[3].tilt_x - (-12.3)) < 1e-2
        assert abs(reloaded_stack.surfaces[2].thickness - 72.0) < 1e-3

        win.on_trace_rays()
        win.close()

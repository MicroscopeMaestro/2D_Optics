"""
Comprehensive Automated Test Suite for Zemax Illumination, Mask & Projection Systems.
====================================================================================
Tests the physical modeling, étendue invariants, radiometric power transfer,
spatial homogenization, pupil matching, LDE conversion, 3D ray tracing,
extended merit operands, GUI dock, and AI tool registry integration.

Theoretical Basis:
- Herbert Gross (Editor), "Handbook of Optical Systems", Wiley-VCH (Vols 2, 4, 6)
- Coursera / University of Colorado Boulder "Optical Engineering"
"""

import os
import math
import pytest
import numpy as np

from optics_app.engine.illumination import (
    IlluminationMaskProjectionTrain,
    IlluminationSource,
    SourceType,
    IlluminationMode,
    HomogenizerType,
    HomogenizerConfig,
    MaskType,
    MaskSubsystemConfig,
    CollectorType,
    ProjectionType,
    ProjectionLensConfig,
    EnergyThroughputMetrics,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.extended_merit import (
    EtendueMatchOperand,
    FluxThroughputOperand,
    IrradianceUniformityOperand,
    TelecentricityOperand,
    PupilMatchingOperand,
)
from optics_app.ai.tools import (
    design_illumination_mask_projection_system,
    calculate_energy_throughput_and_uniformity,
    optimize_pupil_matching_and_throughput,
    get_default_tool_registry,
)


# =============================================================================
# 1. Source Physics & Radiometric Invariants
# =============================================================================

def test_illumination_source_arc_lamp():
    """Verify Arc Lamp emission geometry, solid angle, and geometrical etendue."""
    src = IlluminationSource(
        source_type=SourceType.ARC_LAMP,
        optical_power_w=350.0,
        wavelength_um=0.365,
        emitter_width_mm=1.5,
        emitter_height_mm=1.5,
        numerical_aperture=0.65,
    )
    assert src.emitter_area_mm2 == pytest.approx(2.25, rel=1e-3)
    assert src.solid_angle_sr == pytest.approx(math.pi * (0.65 ** 2), rel=1e-3)
    # G = A * Omega = 2.25 * pi * 0.65^2 ~ 2.986 mm^2*sr
    assert src.etendue_mm2_sr == pytest.approx(2.25 * math.pi * (0.65 ** 2), rel=1e-3)
    assert src.radiance_w_per_mm2_sr > 0.0
    assert 30.0 < src.divergence_half_angle_deg < 50.0


def test_illumination_source_laser_gaussian():
    """Verify Gaussian Laser beam waist, Rayleigh area, and low etendue."""
    src = IlluminationSource(
        source_type=SourceType.LASER_GAUSSIAN,
        optical_power_w=10.0,
        wavelength_um=0.532,
        emitter_width_mm=2.0,  # 2*w0 -> w0 = 1.0 mm
        emitter_height_mm=2.0,
        numerical_aperture=0.015,
        m2_factor=1.1,
    )
    # Area = pi * w0^2 = pi * 1.0^2 = pi
    assert src.emitter_area_mm2 == pytest.approx(math.pi, rel=1e-3)
    # Laser etendue is very small compared to arc lamp
    assert src.etendue_mm2_sr < 0.01
    assert src.radiance_w_per_mm2_sr > 1000.0


# =============================================================================
# 2. Homogenizer Physics & Flat-Top Shaping
# =============================================================================

def test_homogenizer_light_pipe_bounces():
    """Verify internal reflection count and spatial flat-top uniformity in light pipe."""
    homog = HomogenizerConfig(
        homogenizer_type=HomogenizerType.LIGHT_PIPE,
        rod_length_mm=85.0,
        rod_width_mm=16.0,
        rod_height_mm=16.0,
        glass_material="F_SILICA",
        refractive_index=1.474,
        transmission_efficiency=0.95,
    )
    bounces = homog.calculate_number_of_bounces(input_na=0.20)
    assert bounces > 0.5
    u_pct = homog.calculate_spatial_uniformity_pct(input_na=0.20)
    assert 68.0 <= u_pct <= 99.0


def test_homogenizer_flys_eye_and_diffuser():
    """Verify Fly's Eye array provides high spatial averaging."""
    fe = HomogenizerConfig(homogenizer_type=HomogenizerType.FLYS_EYE)
    assert fe.calculate_spatial_uniformity_pct(input_na=0.10) >= 95.0

    raw = HomogenizerConfig(homogenizer_type=HomogenizerType.NONE)
    assert raw.calculate_spatial_uniformity_pct(input_na=0.10) < 70.0


# =============================================================================
# 3. Mask & Projection Subsystems
# =============================================================================

def test_mask_subsystem_geometry():
    """Verify active mask area and bounding semi-diameter."""
    mask = MaskSubsystemConfig(
        mask_type=MaskType.BINARY_RONCHI,
        active_width_mm=20.0,
        active_height_mm=20.0,
        framing_blade_aperture_x_mm=16.0,
        framing_blade_aperture_y_mm=16.0,
    )
    assert mask.active_area_mm2 == pytest.approx(256.0, rel=1e-3)
    # semi-diameter = sqrt(16^2 + 16^2) / 2 ~ 11.31 mm
    assert mask.semi_diameter_mm == pytest.approx(math.sqrt(512.0) / 2.0, rel=1e-3)


def test_projection_lens_resolution_and_dof():
    """Verify Rayleigh resolution, lithography half-pitch, and DOF calculations."""
    proj = ProjectionLensConfig(
        projection_type=ProjectionType.DOUBLE_TELECENTRIC_4X,
        magnification=-0.25,
        image_numerical_aperture=0.45,
        design_wavelength_um=0.365,
    )
    # NA_obj = 0.25 * 0.45 = 0.1125
    assert proj.object_numerical_aperture == pytest.approx(0.1125, rel=1e-3)
    # Rayleigh = 0.61 * 0.365 / 0.45 ~ 0.4947 um
    assert proj.rayleigh_resolution_um == pytest.approx(0.61 * 0.365 / 0.45, rel=1e-3)
    # Litho = 0.40 * 0.365 / 0.45 ~ 0.3244 um
    assert proj.litho_resolution_um == pytest.approx(0.40 * 0.365 / 0.45, rel=1e-3)
    # DOF = 0.365 / (2 * 0.45^2) ~ 0.9012 um
    assert proj.depth_of_focus_um == pytest.approx(0.365 / (2.0 * 0.45**2), rel=1e-3)


# =============================================================================
# 4. Reference Archetypes & Pupil Matching
# =============================================================================

def test_archetype_photolithography_stepper_4x():
    """Verify complete i-Line 365nm 4x reduction photolithography stepper train."""
    train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()
    assert train.source.source_type == SourceType.ARC_LAMP
    assert train.source.wavelength_um == 0.365
    assert train.projection.magnification == -0.25
    assert train.projection.is_double_telecentric is True

    pupil_res = train.solve_pupil_matching()
    assert pupil_res["is_matched"] is True
    assert pupil_res["axial_error_mm"] < 25.0
    assert 0.5 <= pupil_res["partial_coherence_sigma"] <= 1.2

    m = train.evaluate_throughput_and_uniformity()
    assert m.source_power_w == 350.0
    assert m.image_delivered_power_w > 10.0
    assert m.total_optical_throughput_pct > 5.0
    assert m.mask_uniformity_rms_pct >= 90.0
    assert m.image_field_width_mm == pytest.approx(5.0, rel=1e-2)  # 20mm * 0.25 = 5mm


def test_archetype_laser_mask_projection_system():
    """Verify 532nm laser mask projection system with Fly's Eye beam shaper."""
    train = IlluminationMaskProjectionTrain.create_laser_mask_projection_system()
    assert train.source.source_type == SourceType.LASER_GAUSSIAN
    assert train.source.wavelength_um == 0.532
    assert train.homogenizer.homogenizer_type == HomogenizerType.FLYS_EYE

    m = train.evaluate_throughput_and_uniformity()
    assert m.source_power_w == 10.0
    assert m.image_delivered_power_w > 0.5
    assert m.total_optical_throughput_pct > 5.0
    assert m.mask_uniformity_rms_pct >= 95.0


def test_archetype_high_power_lamp_dmd_projector():
    """Verify Xenon 300W lamp DMD projection display system."""
    train = IlluminationMaskProjectionTrain.create_high_power_lamp_projector()
    assert train.source.source_type == SourceType.ARC_LAMP
    assert train.mask.mask_type == MaskType.DMD_REFLECTIVE
    assert train.projection.magnification == 20.0

    m = train.evaluate_throughput_and_uniformity()
    assert m.source_power_w == 300.0
    assert m.image_delivered_power_w > 15.0
    assert m.image_field_width_mm == pytest.approx(280.0, rel=1e-2)  # 14mm * 20 = 280mm


def test_archetype_telecentric_metrology_train():
    """Verify double-telecentric 1:1 inspection and metrology train."""
    train = IlluminationMaskProjectionTrain.create_telecentric_metrology_inspection_system()
    assert train.source.source_type == SourceType.LED
    assert train.projection.magnification == -1.0
    assert train.projection.is_double_telecentric is True

    m = train.evaluate_throughput_and_uniformity()
    assert m.total_optical_throughput_pct > 5.0
    assert m.chief_ray_telecentricity_error_deg < 0.05


# =============================================================================
# 5. Lens Data Editor (LDE) & 3D Ray Tracing Integration
# =============================================================================

def test_surface_stack_conversion_and_ray_tracing():
    """Verify that train converts into SurfaceStack and traces sequential rays."""
    train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()
    stack = train.to_surface_stack()

    assert len(stack.surfaces) >= 15
    assert stack.surfaces[0].surface_type == SurfaceType.OBJECT
    assert stack.surfaces[-1].surface_type == SurfaceType.DETECTOR

    # Verify Stop surface exists inside projection lens
    stop_surfaces = [s for s in stack.surfaces if getattr(s, "is_stop", False)]
    assert len(stop_surfaces) >= 1
    assert "PROJ_STO" in stop_surfaces[0].name

    # Trace 3D rays through OpticalSystem
    opt_sys = OpticalSystem(stack)
    ray_on_axis = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.365)
    traced_on_axis = opt_sys.trace_ray(ray_on_axis)
    assert len(traced_on_axis.history) > 5

    # Trace small cone ray
    ray_cone = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.05, math.sqrt(1.0 - 0.05**2)]), wavelength=0.365)
    traced_cone = opt_sys.trace_ray(ray_cone)
    assert len(traced_cone.history) > 5


# =============================================================================
# 6. Irradiance Profiles Generation (1D & 2D)
# =============================================================================

def test_generate_irradiance_profiles():
    """Verify 1D cross-sectional profiles and 2D spatial irradiance grids."""
    train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()
    profiles = train.generate_irradiance_profiles(grid_size=31)

    assert "mask" in profiles
    assert "image" in profiles
    assert len(profiles["mask"]["slice_1d"]) == 31
    assert len(profiles["image"]["slice_1d"]) == 31
    assert len(profiles["mask"]["grid_2d"]) == 31

    # Check center irradiance is positive and higher than edges
    mid = 15
    assert profiles["mask"]["slice_1d"][mid] > profiles["mask"]["slice_1d"][0]
    assert profiles["image"]["slice_1d"][mid] > profiles["image"]["slice_1d"][0]


# =============================================================================
# 7. Zemax OpticStudio Prescription & Report Export
# =============================================================================

def test_zemax_report_and_zmx_export(tmp_path):
    """Verify ASCII report generation and .zmx file export."""
    train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()
    report = train.generate_radiometric_report()

    assert "ZEMAX OPTICSTUDIO ILLUMINATION & PROJECTION REPORT" in report
    assert "HELMHOLTZ-LAGRANGE INVARIANT & ÉTENDUE THROUGHPUT BUDGET" in report
    assert "RADIOMETRIC POWER TRANSFER & EFFICIENCY AUDIT" in report
    assert "IRRADIANCE & UNIFORMITY METRICS" in report
    assert "PUPIL MATCHING & COHERENCE CONJUGATION" in report

    zmx_file = tmp_path / "stepper_4x.zmx"
    train.export_to_zmx(str(zmx_file))
    assert zmx_file.exists()
    content = zmx_file.read_text(encoding="utf-8")
    assert "VERS 210815" in content
    assert "SURF 0" in content
    assert "GLAS" in content
    assert "COMM" in content


# =============================================================================
# 8. Extended Merit Operands
# =============================================================================

def test_illumination_extended_merit_operands():
    """Verify ETEN, FLUX, UNIF, TELE, and PUMA optimization operands."""
    train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()

    op_eten = EtendueMatchOperand(target_etendue=1.0, weight=1.0)
    val_eten = op_eten.evaluate(train)
    assert val_eten > 0.0

    op_flux = FluxThroughputOperand(target_throughput_pct=20.0, weight=2.0)
    val_flux = op_flux.evaluate(train)
    assert val_flux > 0.0

    op_unif = IrradianceUniformityOperand(target_uniformity_pct=90.0, weight=3.0)
    val_unif = op_unif.evaluate(train)
    assert val_unif >= 85.0

    op_tele = TelecentricityOperand(target_deg=0.0, weight=5.0)
    val_tele = op_tele.evaluate(train)
    assert val_tele < 0.1

    op_puma = PupilMatchingOperand(target_mismatch_mm=0.0, weight=4.0)
    val_puma = op_puma.evaluate(train)
    assert val_puma >= 0.0


# =============================================================================
# 9. AI Tool Registry Integration
# =============================================================================

def test_ai_tool_design_illumination_mask_projection_system(tmp_path):
    """Verify tool execution for designing illumination and projection systems."""
    zmx_out = str(tmp_path / "ai_stepper.zmx")
    res = design_illumination_mask_projection_system(
        archetype="stepper_4x",
        optical_power_w=400.0,
        wavelength_um=0.365,
        push_to_active_system=False,
        export_zmx_path=zmx_out,
    )
    assert res["success"] is True
    assert res["wavelength_um"] == 0.365
    assert res["throughput_metrics"]["source_power_w"] == 400.0
    assert res["throughput_metrics"]["total_optical_throughput_pct"] > 5.0
    assert os.path.exists(zmx_out)


def test_ai_tool_calculate_energy_throughput():
    """Verify tool execution for energy throughput audit."""
    res = calculate_energy_throughput_and_uniformity(archetype="laser_proj")
    assert res["success"] is True
    assert res["system_name"] == "Laser_532nm_Mask_Projection_System"
    assert res["source_power_w"] == 10.0
    assert len(res["mask_1d_center_slice"]) == 31


def test_ai_tool_optimize_pupil_matching():
    """Verify tool execution for pupil matching solve."""
    res = optimize_pupil_matching_and_throughput(archetype="stepper_4x")
    assert res["success"] is True
    assert res["is_matched"] is True
    assert res["axial_error_mm"] < 25.0
    assert "converged" in res["message"]


# =============================================================================
# 10. GUI Dock Initialization & State
# =============================================================================

def test_illumination_dock_gui_initialization():
    """Verify IlluminationDockWidget initializes cleanly and updates state."""
    try:
        from optics_app.gui.qt_compat import QtWidgets, QT_AVAILABLE
        from optics_app.gui.illumination_dock import IlluminationDockWidget
    except ImportError:
        pytest.skip("Qt GUI not available")

    if not QT_AVAILABLE:
        pytest.skip("Qt GUI not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dock = IlluminationDockWidget()

    assert dock is not None
    assert dock.train is not None
    assert dock.table_metrics.rowCount() > 10

    # Test preset switching
    dock.combo_presets.setCurrentIndex(1)  # Laser
    assert dock.train.source.source_type == SourceType.LASER_GAUSSIAN

    dock.combo_presets.setCurrentIndex(2)  # Lamp DMD
    assert dock.train.mask.mask_type == MaskType.DMD_REFLECTIVE

    # Test dark mode toggle
    dock.set_dark_mode(True)
    assert dock.dark_mode is True
    dock.set_dark_mode(False)
    assert dock.dark_mode is False

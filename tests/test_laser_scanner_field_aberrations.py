"""
Comprehensive Test Suite for Laser Scanning System Simulation & Field Aberrations.
Tests:
1. Galvo detection and GalvoScannerConfig mechanical/optical angle math.
2. configure_galvo_scanner surface metadata and mechanical states registration.
3. LaserBeamConfig collimated laser bundle generation and Gaussian weighting.
4. LaserScannerSimulator 1D scan sweep (RMS spot, Airy disk, astigmatism, field curvature, f*theta error).
5. LaserScannerSimulator 2D raster grid scan.
6. AI tools registry and natural language heuristics.
7. GUI LaserScanAberrationTab and AnalysisTabsWidget integration.
8. GUI ComponentEditorDockWidget galvo actuator controls and live angle steering.
"""

import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.laser_scanner import (
    is_galvo,
    configure_galvo_scanner,
    GalvoScannerConfig,
    LaserBeamConfig,
    ScanPointResult,
    LaserScanSimulationResult,
    LaserScannerSimulator,
)
from optics_app.ai.tools import registry
from optics_app.ai.nemotron_client import parse_optical_command_heuristics
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass
from optics_app.gui.analysis_tabs import (
    AnalysisTabsWidget,
    LaserScanAberrationTab,
)
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.lde_table import LensDataTableModel


def create_test_laser_scanning_system() -> SurfaceStack:
    """
    Build a realistic laser scan system:
    Surface 0: Entrance pupil / aperture stop (z=0)
    Surface 1: Galvanometer scanning mirror (tiltable, z=20)
    Surface 2: Scan lens front surface (convex R=50, d=8, N-BK7, z=50)
    Surface 3: Scan lens rear surface (flat R=inf, d=100, Air)
    Surface 4: Detector / workpiece plane (z=158)
    """
    stack = SurfaceStack()
    
    s0 = Surface(name="Entrance Pupil", radius=np.inf, thickness=20.0, glass="AIR", semi_diameter=10.0)
    
    s1 = Surface(
        name="Galvo Mirror X",
        radius=np.inf,
        thickness=30.0,
        glass="MIRROR",
        surface_type=SurfaceType.MIRROR,
        semi_diameter=12.5,
    )
    configure_galvo_scanner(s1, scan_axis="X", min_angle_deg=-10.0, max_angle_deg=10.0, step_size_deg=1.0)

    s2 = Surface(name="Scan Lens Front", radius=50.0, thickness=8.0, glass="N-BK7", semi_diameter=25.0)
    s3 = Surface(name="Scan Lens Rear", radius=np.inf, thickness=95.0, glass="AIR", semi_diameter=25.0)
    s4 = Surface(name="Workpiece Detector", radius=np.inf, thickness=0.0, glass="AIR", semi_diameter=50.0)

    stack.surfaces = [s0, s1, s2, s3, s4]
    stack._update_accumulated_positions()
    return stack


class TestGalvoActuatorAndConfig:
    """Test Galvo configuration, optical doubling, and actuator states."""

    def test_is_galvo_detection(self):
        s_plain = Surface(name="Mirror 1", glass="MIRROR")
        assert not is_galvo(s_plain)

        s_galvo1 = Surface(name="Galvo Mirror X", glass="MIRROR")
        assert is_galvo(s_galvo1)

        s_galvo2 = Surface(name="Scanner Y", glass="MIRROR")
        assert is_galvo(s_galvo2)

        s_mech = Surface(name="M1")
        s_mech.mechanism_type = "GALVO"
        assert is_galvo(s_mech)

    def test_galvo_scanner_config_angles(self):
        cfg = GalvoScannerConfig(min_angle_deg=-8.0, max_angle_deg=8.0, step_size_deg=2.0)
        assert cfg.optical_min_deg == -16.0
        assert cfg.optical_max_deg == 16.0

        mech_angles = cfg.get_mechanical_angles()
        assert len(mech_angles) == 9  # -8, -6, -4, -2, 0, 2, 4, 6, 8
        np.testing.assert_allclose(mech_angles[0], -8.0)
        np.testing.assert_allclose(mech_angles[-1], 8.0)

        opt_angles = cfg.get_optical_angles()
        np.testing.assert_allclose(opt_angles, mech_angles * 2.0)

    def test_configure_galvo_scanner(self):
        surf = Surface(name="Scan Mirror", glass="MIRROR", surface_type=SurfaceType.MIRROR)
        cfg = configure_galvo_scanner(surf, scan_axis="X", min_angle_deg=-12.0, max_angle_deg=12.0, step_size_deg=2.0)

        assert is_galvo(surf)
        assert hasattr(surf, "galvo_config")
        assert surf.galvo_config.scan_axis == "X"
        assert "CENTER" in surf.mechanical_states
        assert "MIN" in surf.mechanical_states
        assert "MAX" in surf.mechanical_states
        assert surf.mechanical_state == "CENTER"


class TestLaserBeamAndRayBundle:
    """Test Laser beam configuration and ray bundle generation."""

    def test_laser_beam_config(self):
        b_cfg = LaserBeamConfig(wavelength_um=0.532, beam_diameter_mm=6.0, num_pupil_rings=3)
        assert b_cfg.beam_radius_mm == 3.0

    def test_bundle_generation(self):
        stack = create_test_laser_scanning_system()
        sim = LaserScannerSimulator(stack)
        b_cfg = LaserBeamConfig(beam_diameter_mm=4.0, num_pupil_rings=2, rays_per_ring=4)
        bundle = sim.generate_collimated_laser_bundle(b_cfg, origin_z=-5.0)

        assert len(bundle) > 1
        chief_ray = bundle[0]
        np.testing.assert_allclose(chief_ray.pos[:2], [0.0, 0.0])
        np.testing.assert_allclose(chief_ray.dir, [0.0, 0.0, 1.0])
        assert chief_ray.intensity == 1.0

        # Peripheral rays should have Gaussian apodization intensity <= 1.0
        for r in bundle[1:]:
            assert 0.0 < r.intensity <= 1.0


class TestLaserScanSimulationEngine:
    """Test 1D and 2D laser scanning simulations and field aberration calculations."""

    def test_simulate_1d_scan(self):
        stack = create_test_laser_scanning_system()
        sim = LaserScannerSimulator(stack)
        b_cfg = LaserBeamConfig(wavelength_um=0.532, beam_diameter_mm=5.0, num_pupil_rings=2, rays_per_ring=6)

        result = sim.simulate_1d_scan(
            scan_axis="X",
            min_angle_deg=-6.0,
            max_angle_deg=6.0,
            step_size_deg=2.0,
            beam_config=b_cfg,
        )

        assert isinstance(result, LaserScanSimulationResult)
        assert result.scan_axis == "X"
        assert len(result.scan_points) == 7  # -6, -4, -2, 0, 2, 4, 6
        assert result.scan_lens_efl_mm > 0.0
        assert result.scan_field_diameter_mm > 0.0

        # Check spot metrics
        assert result.mean_rms_spot_um > 0.0
        assert len(result.airy_radii_um) == 7
        assert np.all(result.airy_radii_um > 0.0)

        # Check field curvature and astigmatism
        assert len(result.tangential_foci_mm) == 7
        assert len(result.sagittal_foci_mm) == 7
        assert len(result.astigmatisms_mm) == 7

        # Center point (0 deg) should have minimum or near-zero field distortion
        center_pt = result.scan_points[3]  # angle 0 deg
        assert abs(center_pt.scan_angle_x_deg) < 1e-6
        assert abs(center_pt.chief_ray_x_mm) < 1.0  # centered
        assert center_pt.f_theta_error_percent == 0.0 or abs(center_pt.f_theta_error_percent) < 15.0

    def test_simulate_2d_raster_scan(self):
        stack = create_test_laser_scanning_system()
        sim = LaserScannerSimulator(stack)
        b_cfg = LaserBeamConfig(wavelength_um=0.532, beam_diameter_mm=4.0, num_pupil_rings=1, rays_per_ring=4)

        result = sim.simulate_2d_raster_scan(
            angle_range_x=(-4.0, 4.0),
            angle_range_y=(-4.0, 4.0),
            step_size_deg=2.0,
            beam_config=b_cfg,
        )

        assert result.scan_axis == "XY"
        # 5 x 5 = 25 points
        assert len(result.scan_points) == 25
        assert result.max_rms_spot_um > 0.0


class TestAIToolsAndHeuristics:
    """Test AI tools registry and natural language parsing for scanning."""

    def test_configure_galvo_scanner_tool(self):
        stack = create_test_laser_scanning_system()
        res = registry.execute(
            "configure_galvo_scanner",
            {
                "surface_index": 1,
                "scan_axis": "X",
                "min_angle_deg": -8.0,
                "max_angle_deg": 8.0,
                "step_size_deg": 1.0,
                "system": stack,
            }
        )
        assert res["status"] == "SUCCESS"
        assert res["surface_index"] == 1
        assert res["scan_axis"] == "X"
        assert res["min_angle_deg"] == -8.0
        assert res["max_angle_deg"] == 8.0

    def test_run_laser_scan_simulation_tool(self):
        stack = create_test_laser_scanning_system()
        res = registry.execute(
            "run_laser_scan_simulation",
            {
                "scan_axis": "X",
                "min_angle_deg": -5.0,
                "max_angle_deg": 5.0,
                "step_size_deg": 2.5,
                "wavelength_um": 0.532,
                "beam_diameter_mm": 5.0,
                "system": stack,
            }
        )
        assert res["status"] == "SUCCESS"
        assert res["scan_axis"] == "X"
        assert res["num_scan_points"] == 5
        assert res["scan_lens_efl_mm"] > 0.0
        assert "mean_rms_spot_um" in res
        assert "max_astigmatism_mm" in res
        assert "max_f_theta_error_percent" in res

    def test_parse_optical_command_heuristics_laser_scan(self):
        h1 = parse_optical_command_heuristics("simulate laser scan +/- 8 deg step 0.5 deg")
        assert h1 is not None
        assert h1["name"] == "run_laser_scan_simulation"
        assert h1["arguments"]["min_angle_deg"] == -8.0
        assert h1["arguments"]["max_angle_deg"] == 8.0
        assert h1["arguments"]["step_size_deg"] == 0.5

        h2 = parse_optical_command_heuristics("configure galvo on surface 2 from -10 to 10 deg")
        assert h2 is not None
        assert h2["name"] == "configure_galvo_scanner"
        assert h2["arguments"]["surface_index"] == 2
        assert h2["arguments"]["min_angle_deg"] == -10.0
        assert h2["arguments"]["max_angle_deg"] == 10.0


class TestGUILaserScanAnalysisTab:
    """Test GUI LaserScanAberrationTab and AnalysisTabsWidget integration."""

    def test_laser_scan_tab_instantiation_and_update(self):
        tab = LaserScanAberrationTab()
        stack = create_test_laser_scanning_system()
        sys = OpticalSystem(stack)

        tab.update_plot(system=sys)
        assert tab.last_result is not None
        assert isinstance(tab.last_result, LaserScanSimulationResult)

    def test_analysis_tabs_widget_contains_laser_scan(self):
        stack = create_test_laser_scanning_system()
        model = LensDataTableModel(stack)
        suite = AnalysisTabsWidget(model=model)

        assert "Laser Scanning & Field Aberrations" in suite.tabs
        assert hasattr(suite, "tab_laser_scan")

        # Switch to the laser scan tab
        assert suite.set_active_tab("Laser Scanning & Field Aberrations")
        assert suite.active_tab == "Laser Scanning & Field Aberrations"
        assert suite.refresh_analysis()


class TestGUIComponentEditorGalvoActuators:
    """Test Component Editor Dock galvo controls and live steering."""

    def test_component_editor_galvo_controls(self):
        stack = create_test_laser_scanning_system()
        model = LensDataTableModel(stack)
        dock = ComponentEditorDockWidget(model=model)

        # Select the galvo mirror (index 1)
        dock.set_selected_surface(1)
        assert hasattr(dock, "galvo_group")
        assert dock.galvo_group.isEnabled()
        assert "Active Galvo" in dock.lbl_galvo_mode.text()

        # Test live steering angle
        dock._set_galvo_live_angle(3.5)
        assert abs(dock.spin_galvo_angle.value() - 3.5) < 1e-4
        assert abs(dock.slider_galvo_angle.value() - 350) <= 1

        # Surface 1 is X-scan so it tilts Y
        surf1 = stack.surfaces[1]
        assert abs(surf1.tilt_y - 3.5) < 1e-4

        # Test step angle
        dock._step_galvo_live_angle(1.0)
        assert abs(surf1.tilt_y - 4.5) < 1e-4

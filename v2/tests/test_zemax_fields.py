"""
Comprehensive Unit Tests for Zemax-Style Paraxial Entrance Pupil Calculation,
Iterative Ray Aiming, and Multi-Field Object Simulation.
"""
import math
import numpy as np
import pytest

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.system_config import (
    SystemConfig, FieldType, FieldPoint, ApertureType
)
from optics_app.engine.pupil_solve import (
    find_stop_surface_index,
    calculate_entrance_pupil,
    aim_ray_to_pupil,
    aim_collimated_ray,
    FIELD_COLORS,
)
from optics_app.analysis.spot import (
    SpotDiagram,
    calculate_spot_metrics,
    generate_spot_diagram_from_system,
)
from optics_app.analysis.rayfan import (
    RayFanAnalysis,
    RayFanData,
)


def _build_test_imaging_system():
    """
    Build a realistic imaging test system:
    S0: Object (thickness=50.0 mm)
    S1: Front Plano-Convex Lens R1=25.0, d=5.0, N-BK7
    S2: Flat exit face R2=inf, d=10.0, AIR
    S3: Aperture Stop (STOP), semi_diameter=4.0, d=15.0, AIR
    S4: Rear Plano-Convex Lens R1=inf, d=5.0, N-BK7
    S5: Curved exit face R2=-30.0, d=35.0, AIR
    S6: Detector / Image plane
    """
    stack = SurfaceStack()
    stack.surfaces.clear()

    # S0: Object
    s0 = Surface(name="Object", radius=np.inf, thickness=50.0, glass="AIR", semi_diameter=25.0)
    # S1: Lens 1 Front
    s1 = Surface(name="L1 Front", radius=25.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
    # S2: Lens 1 Rear
    s2 = Surface(name="L1 Rear", radius=np.inf, thickness=10.0, glass="AIR", semi_diameter=12.5)
    # S3: Aperture Stop
    s3 = Surface(name="Aperture Stop", radius=np.inf, thickness=15.0, glass="AIR", semi_diameter=4.0)
    s3.is_stop = True
    s3.surface_type = SurfaceType.STOP
    # S4: Lens 2 Front
    s4 = Surface(name="L2 Front", radius=np.inf, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
    # S5: Lens 2 Rear
    s5 = Surface(name="L2 Rear", radius=-30.0, thickness=35.0, glass="AIR", semi_diameter=12.5)
    # S6: Detector
    s6 = Surface(name="Detector", radius=np.inf, thickness=0.0, glass="AIR", semi_diameter=15.0)
    s6.surface_type = SurfaceType.DETECTOR

    stack.surfaces.extend([s0, s1, s2, s3, s4, s5, s6])
    stack._update_accumulated_positions()
    return stack


class TestPupilSolveEngine:
    """Test paraxial entrance pupil calculation and ray aiming algorithms."""

    def test_find_stop_surface_index(self):
        stack = _build_test_imaging_system()
        stop_idx = find_stop_surface_index(stack.surfaces)
        assert stop_idx == 3, f"Expected stop index 3, got {stop_idx}"

        # Test single surface sequence fallback
        single_surf = [Surface(name="Flat", radius=np.inf, thickness=10.0)]
        assert find_stop_surface_index(single_surf) == 0

        # Test system without explicit stop falls back to S1
        no_stop = [
            Surface(name="Object", thickness=10.0),
            Surface(name="Lens Front", thickness=5.0),
            Surface(name="Image", thickness=0.0),
        ]
        assert find_stop_surface_index(no_stop) == 1

    def test_calculate_entrance_pupil_location_and_radius(self):
        stack = _build_test_imaging_system()
        z_ep, r_ep, stop_idx = calculate_entrance_pupil(stack.surfaces, wavelength_um=0.5876)

        assert stop_idx == 3
        # S0 thickness is 50.0 mm, so S1 vertex is at Z=50.0.
        # The stop S3 is downstream of S1 (5 mm glass + 10 mm air = 15 mm).
        # Imaged through L1 (R=25.0, t=5.0, N-BK7 n~1.5168), the entrance pupil
        # is a virtual image located near the stop in object space.
        assert z_ep > 50.0, f"Entrance pupil Z {z_ep} should be beyond object plane Z=0"
        assert r_ep > 0.0, f"Entrance pupil radius must be positive, got {r_ep}"
        # Stop semi-diameter is 4.0 mm, so R_ep should be close to 4.0 mm (magnified slightly by L1)
        assert 2.0 <= r_ep <= 8.0, f"Entrance pupil radius {r_ep} outside realistic bounds"

    def test_calculate_entrance_pupil_stop_at_surface_1(self):
        # When S1 is the stop itself, Z_ep is exactly at S1 vertex and R_ep == stop radius
        s0 = Surface(name="Object", thickness=40.0)
        s1 = Surface(name="Stop", thickness=10.0, semi_diameter=6.0)
        s1.is_stop = True
        s1.surface_type = SurfaceType.STOP
        s2 = Surface(name="Lens", thickness=20.0, semi_diameter=12.5)

        z_ep, r_ep, stop_idx = calculate_entrance_pupil([s0, s1, s2])
        assert stop_idx == 1
        assert abs(z_ep - 40.0) < 1e-6
        assert abs(r_ep - 6.0) < 1e-6

    def test_aim_ray_to_pupil_finite_object(self):
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)
        surfaces = stack.surfaces

        # Launch chief ray from on-axis object point (0, 0, 0)
        chief_ray = aim_ray_to_pupil(
            surfaces=surfaces,
            obj_point=(0.0, 0.0, 0.0),
            pupil_norm=(0.0, 0.0),
            wavelength_um=0.5876,
            iterative=True,
            system=system,
        )
        assert abs(chief_ray.pos[0]) < 1e-6
        assert abs(chief_ray.pos[1]) < 1e-6
        assert abs(chief_ray.pos[2]) < 1e-6

        # Trace and verify it hits the center of the aperture stop (S3)
        traced = system.trace_ray(chief_ray)
        assert len(traced.history) > 3
        stop_hit = traced.history[3]
        assert abs(stop_hit["x"]) < 0.05, f"Chief ray X hit at stop was {stop_hit['x']}"
        assert abs(stop_hit["y"]) < 0.05, f"Chief ray Y hit at stop was {stop_hit['y']}"

        # Launch chief ray from off-axis object height: Y = 5.0 mm
        offaxis_chief = aim_ray_to_pupil(
            surfaces=surfaces,
            obj_point=(0.0, 5.0, 0.0),
            pupil_norm=(0.0, 0.0),
            wavelength_um=0.5876,
            iterative=True,
            system=system,
        )
        assert abs(offaxis_chief.pos[1] - 5.0) < 1e-6

        traced_offaxis = system.trace_ray(offaxis_chief)
        stop_hit_off = traced_offaxis.history[3]
        # Iterative aiming should guide chief ray through center of stop within < 0.05 mm
        assert abs(stop_hit_off["y"]) < 0.05, f"Off-axis chief ray Y hit at stop was {stop_hit_off['y']}"

    def test_aim_collimated_ray_infinite_conjugate(self):
        stack = _build_test_imaging_system()
        surfaces = stack.surfaces

        # Launch collimated ray at field angle 3.0 deg along Y
        ray_3deg = aim_collimated_ray(
            surfaces=surfaces,
            field_angle_deg=(0.0, 3.0),
            pupil_norm=(0.0, 0.0),
            wavelength_um=0.5876,
        )

        # Direction should point at +3 degrees in Y: ky = sin(3 deg)
        expected_ky = math.sin(math.radians(3.0))
        assert abs(ray_3deg.dir[1] - expected_ky) < 1e-4
        assert abs(ray_3deg.dir[0]) < 1e-6
        assert ray_3deg.dir[2] > 0.99


class TestSystemConfigMultiField:
    """Test SystemConfig integration with multi-field Zemax parameters."""

    def test_generate_input_rays_multi_field_object_height(self):
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=3.0, weight=1.0),
            FieldPoint(x=0.0, y=6.0, weight=0.5),
        ]
        config.wavelength_list_um = [0.5876]
        config.num_rays = 5

        rays = config.generate_input_rays(
            surfaces=stack.surfaces,
            wavelengths=[0.5876],
            num_rays=5,
            system=system,
        )

        assert len(rays) > 0
        # Check that ray field metadata is correctly assigned
        field_indices = set(r.field_idx for r in rays)
        assert field_indices == {0, 1, 2}

        # Check weights and coords
        f2_rays = [r for r in rays if r.field_idx == 2]
        assert len(f2_rays) > 0
        for r in f2_rays:
            assert abs(r.field_coord[1] - 6.0) < 1e-6
            assert abs(r.field_weight - 0.5) < 1e-6

    def test_generate_input_rays_angle_fields(self):
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.beam_type = "Collimated (Parallel Beam)"
        config.field_type = FieldType.ANGLE_DEG
        config.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=2.5, weight=1.0),
        ]
        config.num_rays = 3

        rays = config.generate_input_rays(
            surfaces=stack.surfaces,
            wavelengths=[0.5876],
            num_rays=3,
            system=system,
        )

        assert len(rays) > 0
        f1_rays = [r for r in rays if r.field_idx == 1]
        assert len(f1_rays) > 0
        expected_ky = math.sin(math.radians(2.5))
        for r in f1_rays:
            assert abs(r.dir[1] - expected_ky) < 1e-3


class TestSpotDiagramMultiField:
    """Test Spot Diagram analysis with multi-field pupil aiming."""

    def test_spot_diagram_per_field_centroid_separation(self):
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0),
            FieldPoint(x=0.0, y=4.0),
        ]

        # Generate on-axis spot
        sd_onaxis = generate_spot_diagram_from_system(
            system=system,
            num_rays=60,
            wavelength_um=0.5876,
            field_point=(0.0, 0.0),
            field_type=FieldType.OBJECT_HEIGHT_MM,
            system_config=config,
        )
        cx0, cy0 = sd_onaxis.calculate_centroid()

        # Generate off-axis spot
        sd_offaxis = generate_spot_diagram_from_system(
            system=system,
            num_rays=60,
            wavelength_um=0.5876,
            field_point=(0.0, 4.0),
            field_type=FieldType.OBJECT_HEIGHT_MM,
            system_config=config,
        )
        cx1, cy1 = sd_offaxis.calculate_centroid()

        # In an imaging system with magnification, off-axis object point Y=4mm must image to non-zero Y
        assert abs(cy1 - cy0) > 0.5, f"Expected significant centroid shift between fields, got {cy0} vs {cy1}"
        assert sd_offaxis.calculate_rms_radius() > 0.0


class TestRayFanMultiField:
    """Test Ray Fan analysis with multi-field chief ray referencing."""

    def test_ray_fan_off_axis_tangential_and_sagittal(self):
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0),
            FieldPoint(x=0.0, y=3.0),
        ]

        analyzer = RayFanAnalysis(py_sampling=21, px_sampling=21)

        fan_data_offaxis = analyzer.trace_ray_fan_from_system(
            system=system,
            wavelength_um=0.5876,
            field_point=(0.0, 3.0),
            field_type=FieldType.OBJECT_HEIGHT_MM,
            system_config=config,
        )

        assert isinstance(fan_data_offaxis, RayFanData)
        assert len(fan_data_offaxis.tangential_ey) == 21
        assert len(fan_data_offaxis.sagittal_ex) == 21

        # Central pupil ray should have zero aberration relative to chief ray
        mid_idx = 10
        assert abs(fan_data_offaxis.tangential_ey[mid_idx]) < 1e-4
        assert abs(fan_data_offaxis.sagittal_ex[mid_idx]) < 1e-4

        # Non-zero pupil boundaries should have aberration
        assert not np.isnan(fan_data_offaxis.tangential_ey[0])
        assert not np.isnan(fan_data_offaxis.tangential_ey[-1])


class TestVisualizer3DMultiField:
    """Test 3D Visualizer volumetric multi-field ray generation and color coding."""

    def test_generate_3d_rays_with_system_config_fields(self):
        from optics_app.gui.visualizer_3d import generate_3d_rays, FIELD_COLORS

        stack = _build_test_imaging_system()
        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0),
            FieldPoint(x=0.0, y=3.0),
            FieldPoint(x=0.0, y=-3.0),
        ]
        config.wavelength_list_um = [0.5876]

        rays_3d = generate_3d_rays(
            surface_stack=stack,
            wavelengths=[0.5876],
            num_rings=1,
            rays_per_ring=4,
            system_config=config,
        )

        assert len(rays_3d) > 0
        field_indices = set(getattr(r, "field_idx", -1) for r in rays_3d)
        assert field_indices == {0, 1, 2}

        # Check color palette
        assert len(FIELD_COLORS) >= 8
        assert FIELD_COLORS[0] == '#2563eb'

    def test_visualizer_3d_widget_color_mode_toggle(self, qtbot=None):
        if not QT_AVAILABLE:
            pytest.skip("Qt not available")
        from optics_app.gui.visualizer_3d import Visualizer3DWidget
        widget = Visualizer3DWidget()
        assert widget.color_mode == "wavelength"
        widget.toggle_color_mode()
        assert widget.color_mode == "field"
        widget.toggle_color_mode()
        assert widget.color_mode == "wavelength"


class TestGUIMultiFieldIntegration:
    """Test GUI Analysis Tabs and Viewers with multi-field Zemax configurations."""

    def test_spot_diagram_tab_multi_field_and_matrix(self):
        if not QT_AVAILABLE:
            pytest.skip("Qt not available")
        from optics_app.gui.analysis_tabs import SpotDiagramTab
        tab = SpotDiagramTab()
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0),
            FieldPoint(x=0.0, y=2.0),
            FieldPoint(x=0.0, y=4.0),
        ]
        config.wavelength_list_um = [0.5876]

        # Update in Matrix Mode (selected_field_idx = -1)
        tab.update_plot(system=system, wavelengths=[0.5876], system_config=config)
        assert hasattr(tab, "combo_field")
        assert tab.combo_field.count() == 4  # All Fields + Field 1, 2, 3
        assert tab.combo_field.itemText(0) == "All Fields (Matrix)"
        assert "Field 1" in tab.combo_field.itemText(1)
        assert "Field 2" in tab.combo_field.itemText(2)
        assert "Field 3" in tab.combo_field.itemText(3)

        # Select Field 2 (data = 1)
        tab.combo_field.setCurrentIndex(2)
        assert tab.selected_field_idx == 1

    def test_ray_fan_tab_multi_field(self):
        if not QT_AVAILABLE:
            pytest.skip("Qt not available")
        from optics_app.gui.analysis_tabs import RayFanTab
        tab = RayFanTab()
        stack = _build_test_imaging_system()
        system = OpticalSystem(stack)

        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0),
            FieldPoint(x=0.0, y=3.0),
        ]

        tab.update_plot(system=system, system_config=config)
        assert hasattr(tab, "combo_field")
        assert tab.combo_field.count() == 2
        assert "Field 1" in tab.combo_field.itemText(0)
        assert "Field 2" in tab.combo_field.itemText(1)

        # Select Field 2
        tab.combo_field.setCurrentIndex(1)
        assert tab.selected_field_idx == 1

    def test_dual_view_color_mode_toggle(self):
        if not QT_AVAILABLE:
            pytest.skip("Qt not available")
        from optics_app.gui.dual_view import DualViewWidget
        widget = DualViewWidget()
        assert widget.color_mode == "wavelength"
        widget.toggle_color_mode()
        assert widget.color_mode == "field"
        widget.toggle_color_mode()
        assert widget.color_mode == "wavelength"

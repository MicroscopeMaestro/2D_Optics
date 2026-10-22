"""
Unit tests for Diffraction Grating reflection/diffraction physics,
single-surface propagation modeling, blazed grating attributes,
and 3D visualizer solid rendering with top ruled face.
"""
import pytest
import numpy as np
import os
import json

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.gui.visualizer_3d import Visualizer3DWidget, QT_AVAILABLE
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def test_surface_blaze_attributes_and_reflection():
    """Verify Surface blaze attributes and is_reflective() behavior for gratings."""
    # 1. Reflective blazed grating (default)
    g_refl = Surface(
        name="Grating_600",
        surface_type=SurfaceType.GRATING,
        grating_lines_per_mm=600.0,
        grating_order=1,
        grating_primary_wavelength_um=1.064,
        grating_blaze_wavelength_um=1.064,
        grating_type="Blazed Reflective",
    )
    assert g_refl.is_reflective() is True
    assert g_refl.is_mirror() is True
    assert g_refl.blaze_wavelength == 1.064
    # Check Littrow blaze angle calculation: sin(theta_b) = (600 * 1.064) / 2000 = 0.3192 => ~18.61 deg
    assert abs(g_refl.blaze_angle - 18.61) < 0.2

    # 2. Transmission grating
    g_trans = Surface(
        name="Transmission_Grating_300",
        surface_type=SurfaceType.GRATING,
        grating_lines_per_mm=300.0,
        grating_order=1,
        grating_primary_wavelength_um=0.6328,
        grating_type="Transmission",
        is_transmission_grating=True,
    )
    assert g_trans.is_reflective() is False
    assert g_trans.is_mirror() is False


def test_single_surface_grating_ray_tracing_physics():
    """Verify that light hits ONE grating surface, suffers physical deviation, and propagates directly."""
    stack = SurfaceStack()
    
    # S0: Object / Source at origin
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=50.0))
    
    # S1: Single-surface Blazed Reflection Grating
    # 600 l/mm, 1064 nm, tilted at -42 deg around X
    stack.add_surface(Surface(
        name="Grating_600",
        surface_type=SurfaceType.GRATING,
        tilt_x=-42.0,
        thickness=30.0,  # Carries propagation directly to next element along diffracted axis
        grating_lines_per_mm=600.0,
        grating_order=1,
        grating_primary_wavelength_um=1.064,
        grating_blaze_wavelength_um=1.064,
        grating_blaze_angle_deg=18.61,
        grating_type="Blazed Reflective",
        glass="MIRROR",
        semi_diameter=25.0,
    ))
    
    # S2: Detector plane at 30 mm downstream
    stack.add_surface(Surface(
        name="Detector",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        semi_diameter=30.0,
    ))
    
    stack.primary_wavelength = 1.064
    stack._update_accumulated_positions()
    
    # Verify surface count is 3 (Source, Grating, Detector) - NO dummy Coordinate Break
    assert len(stack.surfaces) == 3
    
    # Trace chief ray at 1064 nm
    sys_opt = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
    res = sys_opt.trace_ray(ray)
    
    # Must have 3 history points
    assert len(res.history) == 3
    
    h_src, h_grat, h_det = res.history
    assert h_src["vignetted"] is False
    assert h_grat["vignetted"] is False
    assert h_det["vignetted"] is False
    
    # Verify grating hit position
    hit_grat = np.array([h_grat["x_global"], h_grat["y_global"], h_grat["z_global"]])
    np.testing.assert_allclose(hit_grat, [0.0, 0.0, 50.0], atol=1e-5)
    
    # Verify that direction after grating is deviated as expected by grating diffraction equation
    # Grating tilt_x = -42 deg, 600 lp/mm, lambda = 1.064 um
    dir_grat = h_grat["dir"]
    assert dir_grat[0] == pytest.approx(0.0, abs=1e-6)
    assert dir_grat[1] == pytest.approx(-0.692, abs=0.01)
    assert dir_grat[2] == pytest.approx(-0.722, abs=0.01)
    
    # Verify detector hit point matches Grating_pos + thickness * dir_grat
    hit_det = np.array([h_det["x_global"], h_det["y_global"], h_det["z_global"]])
    expected_det = np.array([0.0, 0.0, 50.0]) + 30.0 * dir_grat
    np.testing.assert_allclose(hit_det, expected_det, atol=1e-4)


def test_desktop_spectrometer_single_surface_grating_and_dmd_states():
    """Verify that desktop spectrometer works with single-surface grating and toggles DMD ON/OFF cleanly."""
    filepath = r"C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual.json"
    if not os.path.exists(filepath):
        pytest.skip(f"File {filepath} not found on Desktop")
        
    stack_dict = SurfaceStack.load_from_json(filepath)
    stack = stack_dict["stack"]
    
    # Confirm no CB_After_Grating surface exists
    surface_names = [s.name.lower() for s in stack.surfaces]
    assert "cb_after_grating" not in surface_names, "Model should not have redundant CB_After_Grating"
    
    # Find grating and DMD
    grating_surf = next(s for s in stack.surfaces if s.surface_type == SurfaceType.GRATING)
    dmd_surf = next(s for s in stack.surfaces if "dmd" in s.name.lower())
    
    grating_idx = next(i for i, surface in enumerate(stack.surfaces) if surface is grating_surf)
    following = stack.surfaces[grating_idx + 1]
    assert (grating_surf.thickness > 0.0
            or (following.is_air_space and following.thickness > 0.0)), (
        "Grating spacing may be on the grating or an explicit air-space row"
    )
    assert grating_surf.grating_lines_per_mm == 600.0
    assert grating_surf.grating_primary_wavelength_um == 1.225
    assert grating_surf.grating_blaze_wavelength_um == 1.0
    
    # 1. Test DMD ON: ray must reach detector without vignetting
    dmd_surf.mechanical_state = "ON"
    sys_opt = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
    res_on = sys_opt.trace_ray(ray)
    det_hit_on = res_on.history[-1]
    assert det_hit_on["vignetted"] is False, "Ray must reach detector when DMD is ON"
    
    # 2. Test DMD OFF: ray must be deflected and NOT reach detector unvignetted
    dmd_surf.mechanical_state = "OFF"
    res_off = sys_opt.trace_ray(ray)
    det_hit_off = res_off.history[-1]
    assert det_hit_off["vignetted"] is True, "Ray must be vignetted and blocked from detector when DMD is OFF"


from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

def _ensure_qapp():
    if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return app
    return None


def test_visualizer_3d_grating_top_face_and_rulings():
    """Verify that Visualizer3DWidget renders grating with metallic gold top face, dark substrate, and rulings."""
    _ensure_qapp()
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(Surface(
        name="Ruled_Grating",
        surface_type=SurfaceType.GRATING,
        tilt_x=-42.0,
        thickness=20.0,
        substrate_thickness=6.0,
        substrate_shape="Rectangular",
        aperture_type="Rectangular",
        aperture_x_half=15.0,
        aperture_y_half=12.0,
        grating_lines_per_mm=600.0,
        grating_primary_wavelength_um=1.064,
        grating_blaze_wavelength_um=1.064,
        grating_blaze_angle_deg=18.61,
        grating_type="Blazed Reflective",
    ))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))
    stack._update_accumulated_positions()
    
    widget = Visualizer3DWidget(auto_refresh=False)
    widget.surface_stack = stack
    widget.update_layout()
    
    # Check 3D collections for grating rendering:
    # 1. Poly3DCollection with gold top face (#fbbf24) and gold edge (#b45309)
    # 2. Poly3DCollection with dark substrate (#334155) and dark edge (#1e293b)
    # 3. Ruling lines plotted on ax with dark amber color (#78350f)
    found_gold_face = False
    found_dark_substrate = False
    
    for coll in widget.ax.collections:
        if isinstance(coll, Poly3DCollection):
            fc = coll.get_facecolor()
            ec = coll.get_edgecolor()
            # In matplotlib, facecolors are converted to RGBA floats
            # #fbbf24: [0.984, 0.749, 0.141]
            # #334155: [0.200, 0.255, 0.333]
            if len(fc) > 0:
                r, g, b = fc[0][:3]
                if r > 0.85 and g > 0.65 and b < 0.30:
                    found_gold_face = True
                elif 0.15 < r < 0.25 and 0.20 < g < 0.30 and 0.28 < b < 0.38:
                    found_dark_substrate = True
                    
    assert found_gold_face, "Visualizer3D should render top reflective face in gold"
    assert found_dark_substrate, "Visualizer3D should render substrate blank in dark slate"
    
    # Check that ruling lines exist on axes
    ruling_lines = [
        line for line in widget.ax.lines
        if line.get_color() in ('#78350f', [0.47058823529411764, 0.20784313725490197, 0.058823529411764705, 0.92])
        or (isinstance(line.get_color(), str) and '#78350f' in line.get_color().lower())
    ]
    # At least 11 rulings should be drawn
    assert len(ruling_lines) >= 10, f"Expected at least 10 ruling lines, found {len(ruling_lines)}"
    
    # Check element label text
    # Should contain 'Grating', '600', 'Blazed @ 1064 nm', and '[Top Ruled Face]'
    label_texts = [t.get_text() for t in widget.ax.texts]
    grating_label = next((txt for txt in label_texts if "Grating" in txt), None)
    assert grating_label is not None, "Grating label should be displayed"
    assert "Top Ruled Face" in grating_label
    assert "1064 nm" in grating_label

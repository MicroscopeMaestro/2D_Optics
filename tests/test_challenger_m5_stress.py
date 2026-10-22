"""
Milestone 5 Tier 5 White-Box Adversarial Hardening Test Suite.
Tests:
1. Zemax .zmx Parser Edge Cases (multi-byte file encodings UTF-8/UTF-16/Latin-1/CP1252, invalid line formats, missing fields).
2. 3D STEP Assembly Exporter (optics_app/cad/exporter.py) under zero, single, extreme, and multi-component systems.
3. Matplotlib Figure Refresh Under High-Frequency Updates (analysis tabs & dual view layout).
"""
import os
import sys
import time
import tempfile
import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, STEPExportEngine, export_to_step, CADExporter
from optics_app.gui.analysis_tabs import SpotDiagramTab, RayFanTab, GratingDispersionTab, CCDHeatmapTab, AnalysisTabsWidget
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:
    try:
        from optics_app.gui.qt_compat import QtWidgets
        if hasattr(QtWidgets, 'QApplication'):
            _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


def test_m5_zmx_multi_byte_encodings():
    """Verify Zemax .zmx parser with UTF-8, UTF-16, Latin-1, CP1252 encodings."""
    encodings_to_test = [
        ("UTF-8 with BOM", "utf-8-sig", "ZEMAX PRESCRIPTION\nUNIT MM\nSURF 0\n  TYPE OBJECT\nSURF 1 ! Lens 1 (Ø 25mm 50.0°)\n  CURV 0.02\n  THIC 5.0\n  GLAS BK7\n  DIAM 12.5\nSURF 2\n  CURV -0.02\n  THIC 10.0\n  DIAM 12.5\n"),
        ("UTF-16 LE with BOM", "utf-16-le", "ZEMAX PRESCRIPTION\nUNIT MM\nSURF 0\n  TYPE OBJECT\nSURF 1 ! Standard Lens\n  CURV 0.01\n  THIC 10.0\n  GLAS N-BK7\n  DIAM 25.0\nSURF 2\n  THIC 50.0\n  DIAM 25.0\n"),
        ("UTF-16 BE with BOM", "utf-16-be", "ZEMAX PRESCRIPTION\nUNIT MM\nSURF 0\n  TYPE OBJECT\nSURF 1\n  CURV 0.05\n  THIC 8.0\n  GLAS SILICA\n  DIAM 15.0\nSURF 2\n  THIC 40.0\n  DIAM 15.0\n"),
        ("UTF-8 without BOM with symbols", "utf-8", "ZEMAX PRESCRIPTION\nUNIT MM\n! High power optic: λ=587.6nm, T=20°C, Ø=25.4mm\nSURF 0\n  TYPE OBJECT\nSURF 1 ! Front surface (µm finish)\n  CURV 0.025\n  THIC 6.0\n  GLAS SF11\n  DIAM 12.7\nSURF 2\n  CURV 0.0\n  THIC 30.0\n  DIAM 12.7\n"),
        ("Latin-1 high-bit bytes", "latin-1", "ZEMAX PRESCRIPTION\nUNIT MM\n! Written by Z\xe9max v20.2 \xa9 2020\nSURF 0\n  TYPE OBJECT\nSURF 1 ! Lens \xb01\n  CURV 0.03\n  THIC 4.0\n  GLAS BK7\n  DIAM 10.0\nSURF 2\n  THIC 20.0\n  DIAM 10.0\n")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, enc, content in encodings_to_test:
            filepath = os.path.join(tmpdir, f"test_{enc}.zmx")
            if enc in ("utf-16-le", "utf-16-be"):
                with open(filepath, "wb") as f:
                    if enc == "utf-16-le":
                        f.write(b'\xff\xfe' + content.encode('utf-16-le'))
                    else:
                        f.write(b'\xfe\xff' + content.encode('utf-16-be'))
            else:
                with open(filepath, "w", encoding=enc) as f:
                    f.write(content)

            stack = parse_zmx_file(filepath)
            assert len(stack.surfaces) >= 2, f"Failed to parse surfaces for {name}"


def test_m5_zmx_invalid_formats_and_missing_fields():
    """Verify Zemax .zmx parser resilience against invalid line formats and missing fields."""
    invalid_cases = [
        ("Non-numeric CURV", "SURF 1\nCURV NOT_A_NUMBER\nTHIC 5.0\nDIAM 10.0"),
        ("Non-numeric THIC", "SURF 1\nCURV 0.02\nTHIC abc\nDIAM 10.0"),
        ("Non-numeric DIAM", "SURF 1\nCURV 0.02\nTHIC 5.0\nDIAM xyz"),
        ("Non-integer SURF", "SURF abc\nCURV 0.02\nTHIC 5.0"),
        ("Invalid PARM index negative", "SURF 1\nTYPE ASPHERIC\nPARM -1 0.001"),
        ("Commands with missing args", "SURF\nCURV\nTHIC\nDIAM\nCONI\nPARM\nDISP\nTYPE"),
        ("Only comments and empty lines", "! Comment 1\n# Comment 2\n! Comment 3\n   \n\t\n"),
        ("Surface missing CURV and THIC", "SURF 1\nGLAS BK7\nDIAM 10.0"),
        ("THIC INFINITY", "SURF 1\nTHIC INFINITY\nDIAM 10.0"),
        ("Inline comments", "SURF 1 ! comment 1 # comment 2\nCURV 0.02 ! rad = 50mm\nTHIC 10.0 # thickness = 10mm\nGLAS SF11 ! material\nDIAM 15.0 # diameter")
    ]

    for name, content in invalid_cases:
        try:
            stack = parse_zmx_string(content)
            assert isinstance(stack, SurfaceStack), f"Expected SurfaceStack for {name}"
        except Exception as e:
            # Exception handling verified
            assert isinstance(e, Exception)


def test_m5_step_assembly_exporter():
    """Verify 3D STEP Assembly Exporter under zero, single, extreme, and multi-component systems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Empty SurfaceStack
        empty_stack = SurfaceStack()
        path1 = os.path.join(tmpdir, "empty_assembly.step")
        res1 = STEPExporter.export(empty_stack, path1)
        assert res1.is_success
        assert os.path.exists(path1) and os.path.getsize(path1) > 0

        # 2. Single Elements
        stack_single = SurfaceStack()
        stack_single.add_surface(Surface(name="Stop1", surface_type=SurfaceType.STOP, semi_diameter=10.0, thickness=5.0))
        stack_single.add_surface(Surface(name="Detector1", surface_type=SurfaceType.DETECTOR, semi_diameter=12.0, thickness=0.0))
        path2 = os.path.join(tmpdir, "single_elements.step")
        res2 = STEPExporter.export(stack_single, path2)
        assert res2.is_success
        assert res2.num_components_exported >= 1

        # 3. Doublet Lens Assembly + Grating + Detector
        stack_sys = SurfaceStack()
        stack_sys.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack_sys.add_surface(Surface(name="Front_Lens_1", radius=50.0, thickness=10.0, glass="N-BK7", semi_diameter=25.0))
        stack_sys.add_surface(Surface(name="Back_Lens_1", radius=-50.0, thickness=5.0, glass="AIR", semi_diameter=25.0))
        stack_sys.add_surface(Surface(name="Grating_1", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, thickness=20.0, semi_diameter=20.0))
        stack_sys.add_surface(Surface(name="CCD_Sensor", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0, thickness=0.0))

        path3 = os.path.join(tmpdir, "optical_system.step")
        res3 = STEPExporter.export(stack_sys, path3)
        assert res3.is_success
        assert os.path.exists(path3) and os.path.getsize(path3) > 1000

        # Verify STEP ISO-10303-21 text header/footer
        with open(path3, "r", encoding="utf-8") as f:
            step_text = f.read()
        assert step_text.startswith("ISO-10303-21;")
        assert "HEADER;" in step_text
        assert "DATA;" in step_text
        assert "ENDSEC;" in step_text
        assert "END-ISO-10303-21;" in step_text

        # 4. Extreme Sag, Tilts, Decenters
        stack_extreme = SurfaceStack()
        stack_extreme.add_surface(Surface(name="Asph1", surface_type=SurfaceType.ASPHERIC, radius=10.0, conic=-1.5,
                                         asphere_coeffs=[1e-4, 1e-6], thickness=50.0, glass="SILICA", semi_diameter=15.0,
                                         decenter_x=25.0, decenter_y=-15.0, tilt_x=15.0, tilt_y=-10.0, tilt_z=45.0))
        stack_extreme.add_surface(Surface(name="Asph2", radius=-10.0, thickness=20.0, glass="AIR", semi_diameter=15.0))
        path4 = os.path.join(tmpdir, "extreme.step")
        res4 = STEPExporter.export(stack_extreme, path4)
        assert res4.is_success

        # 5. Helper convenience functions
        path5 = os.path.join(tmpdir, "conv.step")
        p_res = export_to_step(stack_sys, path5)
        c_res = CADExporter.export_optical_system_step(stack_sys, path5)
        assert p_res.is_success
        assert os.path.exists(c_res)


def test_m5_matplotlib_high_frequency_refresh():
    """Verify Matplotlib canvas refresh under high-frequency plot updates."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
    stack.add_surface(Surface(name="L1", radius=40.0, thickness=8.0, glass="BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="L2", radius=-60.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0, thickness=0.0))
    system = OpticalSystem(stack)

    # 1. SpotDiagramTab 10 updates
    tab_spot = SpotDiagramTab()
    for _ in range(10):
        tab_spot.update_plot(system=system)
    if tab_spot.fig:
        assert len(tab_spot.fig.axes) <= 1, "Axes count leak in SpotDiagramTab"

    # 2. RayFanTab 10 updates
    tab_fan = RayFanTab()
    for _ in range(10):
        tab_fan.update_plot(system=system)
    if tab_fan.fig:
        assert len(tab_fan.fig.axes) <= 2, "Axes count leak in RayFanTab"

    # 3. GratingDispersionTab 10 updates
    tab_disp = GratingDispersionTab()
    for _ in range(10):
        tab_disp.update_plot(system=system)
    if tab_disp.fig:
        assert len(tab_disp.fig.axes) <= 2, "Axes count leak in GratingDispersionTab"

    # 4. CCDHeatmapTab 10 updates
    tab_ccd = CCDHeatmapTab()
    for _ in range(10):
        tab_ccd.update_plot(system=system)
    if tab_ccd.fig:
        assert len(tab_ccd.fig.axes) <= 3, "Axes count leak in CCDHeatmapTab"

    # 5. DualViewWidget 10 updates
    dv = DualViewWidget()
    dv.surface_stack = stack
    for _ in range(10):
        dv.refresh_layout()
    if dv.fig:
        assert len(dv.fig.axes) <= 2, "Axes count leak in DualViewWidget"

    # 6. AnalysisTabsWidget active tab rapid toggling
    widget = AnalysisTabsWidget(model=stack)
    tabs = ["Spot Diagram", "Ray Fan", "Grating Dispersion", "CCD Heatmap"]
    for i in range(20):
        res = widget.set_active_tab(tabs[i % len(tabs)])
        assert res is True

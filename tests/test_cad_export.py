"""
Unit tests for 3D STEP Assembly Exporter (Feature 20).
"""
import os
import pytest
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import CADExporter

def test_export_system_step_creates_file(tmp_path):
    stack = SurfaceStack()
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0)
    s1 = Surface(name="Lens1_Front", surface_type=SurfaceType.ASPHERIC, radius=50.0, conic=-1.0, thickness=5.0, glass="N-BK7")
    s2 = Surface(name="Lens1_Back", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=20.0)
    s3 = Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, tilt_x=10.0, thickness=15.0)
    s4 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)
    
    stack.add_surface(s0)
    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)
    stack.add_surface(s4)

    export_path = os.path.join(tmp_path, "test_system.step")
    result_path = CADExporter.export_system_step(stack, export_path)

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0

def test_step_file_valid_structure(tmp_path):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Lens", radius=100.0, thickness=10.0, glass="N-BK7"))
    stack.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR))

    export_path = os.path.join(tmp_path, "structure.step")
    CADExporter.export_system_step(stack, export_path)

    with open(export_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert content.startswith("ISO-10303-21;")
    assert "HEADER;" in content
    assert "DATA;" in content
    assert "ENDSEC;" in content
    assert content.strip().endswith("END-ISO-10303-21;")

def test_export_system_with_tilts_and_decenters(tmp_path):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="TiltedLens", radius=50.0, thickness=8.0, glass="N-SF11", tilt_x=15.0, decenter_x=2.0))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=1200.0, tilt_y=-5.0, thickness=30.0))
    stack.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR))

    export_path = os.path.join(tmp_path, "tilted_system.stp")
    CADExporter.export_system_step(stack, export_path)

    with open(export_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert "Element_1_TiltedLens" in content
    assert "Element_2_Grating" in content
    assert "MANIFOLD_SOLID_BREP" in content

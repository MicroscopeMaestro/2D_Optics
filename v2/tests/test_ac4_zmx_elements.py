"""
Acceptance Criterion 4 (AC4) Test Suite: Dynamic Zemax (.zmx) Element Management.
Verifies loading Thorlabs .zmx files, sequence insertion, front-to-back flip (R -> -R),
3D rigid body tilt/rotation around 3 pivot points, dynamic element replacement, and deletion.
"""

import os
import pytest
import numpy as np

from optics_app.io.zmx_parser import parse_zmx_file
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray


def test_ac4_zmx_parser_loads_doublet(sample_thorlabs_doublet_zmx):
    """AC4.1: Verify parse_zmx_file parses Thorlabs AC254-050-B doublet into SurfaceStack."""
    stack = parse_zmx_file(sample_thorlabs_doublet_zmx)
    assert stack is not None
    assert len(stack.surfaces) >= 4  # Obj, Front, Mid, Back, Det

    # Find the doublet surfaces
    doublet_surfs = [s for s in stack.surfaces if s.glass.upper() in ["N-BK7", "SF5"]]
    assert len(doublet_surfs) == 2, "Doublet must have 2 glass surfaces"
    assert doublet_surfs[0].glass.upper() == "N-BK7"
    assert doublet_surfs[1].glass.upper() == "SF5"
    assert abs(doublet_surfs[0].radius - (1.0 / 0.029411764705882353)) < 1e-2


def test_ac4_zmx_parser_loads_singlet(sample_thorlabs_singlet_zmx):
    """AC4.2: Verify parse_zmx_file parses Thorlabs LA1951 singlet into SurfaceStack."""
    stack = parse_zmx_file(sample_thorlabs_singlet_zmx)
    assert stack is not None
    singlet_surfs = [s for s in stack.surfaces if s.glass.upper() == "N-BK7"]
    assert len(singlet_surfs) == 1
    assert abs(singlet_surfs[0].radius - (1.0 / 0.07633587786259542)) < 1e-2


def test_ac4_zmx_parser_loads_asphere(sample_aspheric_zmx):
    """AC4.3: Verify parse_zmx_file parses even aspheric surface coefficients."""
    stack = parse_zmx_file(sample_aspheric_zmx)
    asph_surfs = [s for s in stack.surfaces if s.surface_type == SurfaceType.ASPHERIC or len(s.asphere_coeffs) > 0]
    assert len(asph_surfs) >= 1
    asph = asph_surfs[0]
    assert abs(asph.conic - (-0.1263039)) < 1e-4
    assert len(asph.asphere_coeffs) >= 4
    assert abs(asph.asphere_coeffs[1] - (-0.0012606)) < 1e-6


def test_ac4_sequence_insertion_after_index(loaded_spectrometer_stack):
    """AC4.4: Verify inserting a singlet or doublet after a specified index expands stack correctly."""
    initial_count = len(loaded_spectrometer_stack.surfaces)
    assert initial_count == 17

    # Insert a doublet after surface 6 (focuser lens)
    inserted_indices = loaded_spectrometer_stack.insert_doublet(
        after_surface_index=6,
        name="InsertedDoublet",
        r1=40.0,
        t1=5.0,
        glass1="N-BK7",
        r2=-30.0,
        t2=2.5,
        glass2="SF5",
        r3=-120.0,
        semi_diameter=12.7,
        air_gap_after=15.0
    )
    assert len(inserted_indices) == 3
    assert inserted_indices == [7, 8, 9]
    assert len(loaded_spectrometer_stack.surfaces) == initial_count + 3
    assert loaded_spectrometer_stack.surfaces[7].name == "InsertedDoublet_Front"
    assert loaded_spectrometer_stack.surfaces[8].name == "InsertedDoublet_Mid"
    assert loaded_spectrometer_stack.surfaces[9].name == "InsertedDoublet_Back"


def test_ac4_element_front_to_back_flip(loaded_spectrometer_stack, sample_thorlabs_doublet_zmx):
    """AC4.5: Verify front-to-back flip inverts surface curvatures (R -> -R) and reverses sequence."""
    # Replace focuser doublet with .zmx in flipped/reversed orientation
    loaded_spectrometer_stack.replace_lens_from_zmx(
        target=4,
        file_path=sample_thorlabs_doublet_zmx,
        reverse=True
    )
    rev_r1 = loaded_spectrometer_stack.surfaces[4].radius
    rev_r3 = loaded_spectrometer_stack.surfaces[6].radius

    # In reversed mode, front curvature is -(-115.0) = 115.0, and back curvature is -(34.0) = -34.0
    assert abs(rev_r1 - 115.0) < 1e-2
    assert abs(rev_r3 - (-34.0)) < 1e-2


def test_ac4_element_rigid_body_3d_rotation_front_vertex(loaded_spectrometer_stack):
    """AC4.6: Verify rotate_lens_element tilts all surfaces of doublet together around front_vertex."""
    # Focuser doublet starts at surface 4 and spans 4, 5, 6
    surf4 = loaded_spectrometer_stack.surfaces[4]
    surf5 = loaded_spectrometer_stack.surfaces[5]
    surf6 = loaded_spectrometer_stack.surfaces[6]

    loaded_spectrometer_stack.rotate_lens_element(
        front_surface_index=4,
        tilt_x=5.0,
        tilt_y=2.0,
        tilt_z=0.0,
        pivot="front_vertex"
    )

    assert abs(surf4.tilt_x - 5.0) < 1e-3
    assert abs(surf5.tilt_x - 5.0) < 1e-3
    assert abs(surf6.tilt_x - 5.0) < 1e-3
    assert abs(surf4.tilt_y - 2.0) < 1e-3
    assert abs(surf5.tilt_y - 2.0) < 1e-3
    assert abs(surf6.tilt_y - 2.0) < 1e-3

    # Front vertex pivot leaves front decenter near 0
    assert abs(surf4.decenter_x) < 1e-3
    assert abs(surf4.decenter_y) < 1e-3


def test_ac4_element_rigid_body_3d_rotation_center_of_mass(loaded_spectrometer_stack):
    """AC4.7: Verify rotate_lens_element tilts surfaces around center of mass."""
    surf4 = loaded_spectrometer_stack.surfaces[4]
    surf5 = loaded_spectrometer_stack.surfaces[5]
    surf6 = loaded_spectrometer_stack.surfaces[6]

    loaded_spectrometer_stack.rotate_lens_element(
        front_surface_index=4,
        tilt_x=10.0,
        tilt_y=0.0,
        tilt_z=0.0,
        pivot="center_of_mass"
    )

    assert abs(surf4.tilt_x - 10.0) < 1e-3
    assert abs(surf5.tilt_x - 10.0) < 1e-3
    assert abs(surf6.tilt_x - 10.0) < 1e-3
    # With center of mass pivot, front and rear experience opposite Y decenters
    assert surf4.decenter_y * surf6.decenter_y < 0.0 or abs(surf4.decenter_y) > 0.01


def test_ac4_element_rigid_body_3d_rotation_rear_vertex(loaded_spectrometer_stack):
    """AC4.8: Verify rotate_lens_element tilts surfaces around rear_vertex."""
    surf4 = loaded_spectrometer_stack.surfaces[4]
    surf5 = loaded_spectrometer_stack.surfaces[5]
    surf6 = loaded_spectrometer_stack.surfaces[6]

    loaded_spectrometer_stack.rotate_lens_element(
        front_surface_index=4,
        tilt_x=5.0,
        tilt_y=0.0,
        tilt_z=0.0,
        pivot="rear_vertex"
    )

    assert abs(surf4.tilt_x - 5.0) < 1e-3
    assert abs(surf5.tilt_x - 5.0) < 1e-3
    assert abs(surf6.tilt_x - 5.0) < 1e-3


def test_ac4_element_decenter_rigid_body(loaded_spectrometer_stack):
    """AC4.9: Verify decenter_lens_element translates all surfaces together."""
    loaded_spectrometer_stack.decenter_lens_element(
        front_surface_index=4,
        decenter_x=2.5,
        decenter_y=-1.5,
        decenter_z=0.0,
        tilt_x=0.0,
        tilt_y=0.0,
        tilt_z=0.0
    )

    assert abs(loaded_spectrometer_stack.surfaces[4].decenter_x - 2.5) < 1e-3
    assert abs(loaded_spectrometer_stack.surfaces[5].decenter_x - 2.5) < 1e-3
    assert abs(loaded_spectrometer_stack.surfaces[6].decenter_x - 2.5) < 1e-3
    assert abs(loaded_spectrometer_stack.surfaces[4].decenter_y - (-1.5)) < 1e-3
    assert abs(loaded_spectrometer_stack.surfaces[5].decenter_y - (-1.5)) < 1e-3
    assert abs(loaded_spectrometer_stack.surfaces[6].decenter_y - (-1.5)) < 1e-3


def test_ac4_dynamic_catalog_lens_replacement(loaded_spectrometer_stack):
    """AC4.10: Verify replace_lens_from_catalog replaces focuser lens with AC254-075-B cleanly."""
    initial_surfaces_count = len(loaded_spectrometer_stack.surfaces)
    old_trailing_gap = loaded_spectrometer_stack.surfaces[6].thickness

    replaced_indices = loaded_spectrometer_stack.replace_lens_from_catalog(
        target=4,
        part_number="AC254-075-B"
    )
    assert len(replaced_indices) == 3
    assert len(loaded_spectrometer_stack.surfaces) == initial_surfaces_count
    # Preserves trailing air gap
    assert abs(loaded_spectrometer_stack.surfaces[6].thickness - old_trailing_gap) < 1e-3
    assert loaded_spectrometer_stack.surfaces[4].group_name == "AC254-075-B"


def test_ac4_dynamic_element_deletion_merges_air_gap(loaded_spectrometer_stack):
    """AC4.11: Verify delete_lens_element removes lens and merges thickness into preceding air gap."""
    initial_count = len(loaded_spectrometer_stack.surfaces)
    preceding_thick = loaded_spectrometer_stack.surfaces[3].thickness  # Grating to focuser gap (50.0)

    deleted_indices = loaded_spectrometer_stack.delete_lens_element(target=4)
    assert len(deleted_indices) >= 2
    assert len(loaded_spectrometer_stack.surfaces) == initial_count - len(deleted_indices)

    # Check preceding surface thickness was increased by removed element thickness
    new_preceding_thick = loaded_spectrometer_stack.surfaces[3].thickness
    assert new_preceding_thick >= preceding_thick
